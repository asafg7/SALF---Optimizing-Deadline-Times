import copy
import gc
import random
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from statistics import mean
from torchinfo import summary
from collections import OrderedDict
from itertools import islice
from numpy.random import randint

from configurations import args_parser
import utils
import models
from optimal_deadlines import get_optimal_deadlines, get_optimal_deadlines_batchsize

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    textio, best_val_acc, path_best_model = utils.initializations(args)
    textio.cprint(str(args))

    # data
    train_data, test_loader = utils.data(args)
    input, output, train_data, val_loader = utils.data_split(train_data, len(test_loader.dataset), args)

    num_layers, layer_to_stop_arr = utils.get_layers(args.model)
    N_samples = utils.get_samples_count(args.data)
    P_u = np.random.uniform(low=0.0, high=args.maxR, size=args.num_users)
    B_u = np.random.uniform(low=0.0, high=args.maxB, size=args.num_users)

    if args.deadline_times == "uniform" or args.eval:
        iteration_times = np.ones(args.global_epochs)*args.t_max/args.global_epochs
        m_factor = 1
    else:
        iters = np.arange(1, args.global_epochs + 1)
        kappa = args.rho_s / args.rho_c
        l_gamma = np.max((8 * kappa, 1)) - 1

        if args.deadline_times == "fixed":
            etta = args.lr * np.ones(np.size(iters))
        elif args.deadline_times == "inverse":
            etta = args.lr / (iters + 1)

        if args.batchsize_optimization:
            iteration_times, m_factor = get_optimal_deadlines_batchsize(args.num_users, num_layers, args.global_epochs,
                                                              args.t_max, args.g, args.rho_s, args.rho_c, args.gamma,
                                                              args.t_min, args.mean_std, etta, args.alpha, N_samples,
                                                                        args.train_batch_size, P_u)
        else:
            iteration_times = get_optimal_deadlines(args.num_users, num_layers, args.global_epochs,
                                                              args.t_max, args.g, args.rho_s, args.rho_c, args.gamma,
                                                              args.t_min, etta, P_u)
            m_factor = 1
    time_factor = (iteration_times[0] - B_u) / iteration_times[0]
    time_factor = np.maximum(0, time_factor)  # Ensure no negative batch sizes
    train_batch_size = np.round(m_factor * P_u * time_factor).astype(int)
    np.save(f'checkpoints/{args.exp_name}/iteration_times.npy', iteration_times)

    N_iterations = args.monte_carlo_iterations
    mc_array = range(N_iterations)
    train_loss_mat = np.zeros((N_iterations, args.global_epochs))
    val_acc_mat = np.zeros((N_iterations, args.global_epochs))

    for n_itr in mc_array:
        # model
        if args.model == 'mlp':
            global_model = models.FC2Layer(input, output)
        elif args.model == 'cnn2':
            global_model = models.CNN2Layer(input, output, args.data)
        elif args.model[:3] == 'VGG':
            if args.data != 'cifar10':
                raise AssertionError('for VGG data must be cifar10')
            global_model = models.VGG(args.model)
        else:
            AssertionError('invalid model')
        textio.cprint(str(summary(global_model, verbose=0)))
        global_model.to(args.device)

        train_creterion = torch.nn.CrossEntropyLoss(reduction='mean')
        test_creterion = torch.nn.CrossEntropyLoss(reduction='sum')

        #  inference
        if args.eval:
            global_model.load_state_dict(torch.load(path_best_model))
            test_acc = utils.test(test_loader, global_model, test_creterion, args.device)
            textio.cprint(f'eval test_acc: {test_acc:.0f}%')
            gc.collect()
            sys.exit()

        local_models = utils.federated_setup(global_model, train_data, args, train_batch_size)

        # stragglers
        num_of_layers = global_model.state_dict().keys().__len__()
        if args.stragglers is not None:
            stragglers_idx = random.sample(range(args.num_users), round(args.stragglers_percent * args.num_users))
        else:
            stragglers_idx = []

        for idx, global_epoch in tqdm(enumerate(range(0, args.global_epochs))):
            utils.distribute_model(local_models, global_model)
            users_loss = []
            for user_idx in range(args.num_users):
                if (args.stragglers == 'drop') & (user_idx in stragglers_idx) &\
                        (np.random.poisson(iteration_times[global_epoch]) < num_layers):
                    user_new_state_dict = copy.deepcopy(global_model).state_dict()
                    user_new_state_dict.update({})
                    local_models[user_idx]['model'].load_state_dict(user_new_state_dict)
                    continue

                user_loss = []
                for local_epoch in range(0, args.local_epochs):
                    user = local_models[user_idx]
                    train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'], user['scheduler'],
                                                       train_creterion, args.device, args.local_iterations)
                    user_loss.append(train_loss)

                if (args.stragglers == 'adel-fl') & (user_idx in stragglers_idx):
                    user_new_state_dict = copy.deepcopy(global_model).state_dict()
                    if args.up_to_layer is not None:
                        up_to_layer = num_of_layers - args.up_to_layer  # last-to-first layers updated
                    else:
                        poiss_val = np.minimum(np.random.poisson(iteration_times[global_epoch]/m_factor), num_layers)
                        up_to_layer = layer_to_stop_arr[poiss_val]
                    a = user['model'].state_dict()
                    user_updated_layers = OrderedDict(islice(reversed(user['model'].state_dict().items()), up_to_layer))
                    user_new_state_dict.update(user_updated_layers)
                    user['model'].load_state_dict(user_new_state_dict)

                try:
                    users_loss.append(mean(user_loss))
                except:
                    continue
            try:
                train_loss = mean(users_loss)
            except:
                train_loss = 0
            utils.FedAvg(local_models, global_model)

            val_acc = utils.test(val_loader, global_model, test_creterion, args.device)
            train_loss_mat[n_itr, idx] = train_loss
            val_acc_mat[n_itr, idx] = val_acc

            gc.collect()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(global_model.state_dict(), path_best_model)

            textio.cprint(f'epoch: {global_epoch} | train_loss: {train_loss:.2f} | val_acc: {val_acc:.0f}%')

        mean_loss = np.mean(train_loss_mat, axis=0)
        mean_val_acc = np.mean(val_acc_mat, axis=0)
        np.save(f'checkpoints/{args.exp_name}/train_loss_list.npy', mean_loss)
        np.save(f'checkpoints/{args.exp_name}/val_acc_list.npy', mean_val_acc)
        np.save(f'checkpoints/{args.exp_name}/args.npy', args)

        elapsed_min = (time.time() - start_time) / 60
        textio.cprint(f'total execution time: {elapsed_min:.0f} min')
