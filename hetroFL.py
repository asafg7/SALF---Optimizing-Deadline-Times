import copy
import gc
import math
import random
import sys
import time
import torch
import numpy as np
from torch import optim
from torchinfo import summary
from tqdm import tqdm
from statistics import mean

from configurations import args_parser
import utils
import models


def hetroFL_federated_setup():
    if args.lr_decay == "fixed":
        lambda_func = lambda epoch: args.lr
    elif args.lr_decay == "inverse":
        lambda_func = lambda epoch: args.lr / (1 + epoch)
    # create a dict of dict s (local users), i.e. {'1': {'data':..., 'model':..., 'opt':...}, ...}
    indexes = torch.randperm(len(train_data))
    user_data_len = math.floor(len(train_data) / args.num_users) if args.num_samples == None else args.num_samples
    local_models = {}
    for user_idx in range(args.num_users):
        user = {'data': torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data,
                                    indexes[user_idx * user_data_len:(user_idx + 1) * user_data_len]),
            batch_size=args.train_batch_size, shuffle=True),
            'model': copy.deepcopy(global_model) if user_idx in non_stragglers_idx else copy.deepcopy(intermidiate_global_model)}
        user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                                momentum=args.momentum) if args.optimizer == 'sgd' \
            else optim.Adam(user['model'].parameters(), lr=args.lr)
        user['scheduler'] = optim.lr_scheduler.LambdaLR(user['opt'], lr_lambda=lambda_func)
        local_models[user_idx] = user
    return local_models


def hetroFL_distribute_model(mlp_inter_size_1=16, mlp_inter_size_2=8, cnn2_inter_size_1=3, cnn2_inter_size_2=25, cnn2_kernel_size=3):
    for user_idx in non_stragglers_idx:
        local_models[user_idx]['model'].load_state_dict(global_model.state_dict())

    for user_idx in stragglers_idx:
        if args.model == 'mlp':
            local_models[user_idx]['model'].fc1.weight.data = global_model.fc1.weight.data[:mlp_inter_size_1, :]  # weight matrix is [output_size, input_size]
            local_models[user_idx]['model'].fc1.bias.data = global_model.fc1.bias.data[:mlp_inter_size_1]

            local_models[user_idx]['model'].fc2.weight.data = global_model.fc2.weight.data[:mlp_inter_size_2, :mlp_inter_size_1]
            local_models[user_idx]['model'].fc2.bias.data = global_model.fc2.bias.data[:mlp_inter_size_2]

            local_models[user_idx]['model'].fc3.weight.data = global_model.fc3.weight.data[:, :mlp_inter_size_2]
            local_models[user_idx]['model'].fc3.bias.data = global_model.fc3.bias.data

        elif args.model == 'cnn2':
            local_models[user_idx]['model'].conv1.weight.data = global_model.conv1.weight.data[:cnn2_inter_size_1, :, :cnn2_kernel_size, :cnn2_kernel_size]  # [out_channels, in_channels, kernel_size[0], kernel_size[1]]
            local_models[user_idx]['model'].conv1.bias.data = global_model.conv1.bias.data[:cnn2_inter_size_1]

            local_models[user_idx]['model'].conv2.weight.data = global_model.conv2.weight.data[:cnn2_inter_size_1, :cnn2_inter_size_1, :cnn2_kernel_size, :cnn2_kernel_size]
            local_models[user_idx]['model'].conv2.bias.data = global_model.conv2.bias.data[:cnn2_inter_size_1]

            local_models[user_idx]['model'].fc1.weight.data = global_model.fc1.weight.data[:cnn2_inter_size_2, :cnn2_inter_size_1 * data_size * data_size]
            local_models[user_idx]['model'].fc1.bias.data = global_model.fc1.bias.data[:cnn2_inter_size_2]

            local_models[user_idx]['model'].fc2.weight.data = global_model.fc2.weight.data[:, :cnn2_inter_size_2]  # [output_size, input_size]
            local_models[user_idx]['model'].fc2.bias.data = global_model.fc2.bias.data
        else:
            raise AssertionError('not a valid model')


def hetroFL_FedAvg():
    state_dict = global_model.state_dict()
    for key in state_dict.keys():
        local_weights_sum = torch.zeros_like(state_dict[key])

        for user_idx in non_stragglers_idx:
            local_weights_sum += local_models[user_idx]['model'].state_dict()[key]

        if stragglers_idx:
            local_weights_sum_stragglers = torch.zeros_like(intermidiate_global_model.state_dict()[key])
            for user_idx in stragglers_idx:
                local_weights_sum_stragglers += local_models[user_idx]['model'].state_dict()[key]

            # here, pad = (padding_left, padding_right, padding_top, padding_bottom,padding_front, padding_back)
            if len(local_weights_sum_stragglers.shape) > 2:
                padding = torch.nn.ConstantPad3d(
                    padding=(0, local_weights_sum.shape[3] - local_weights_sum_stragglers.shape[3],
                             0, local_weights_sum.shape[2] - local_weights_sum_stragglers.shape[2],
                             0, local_weights_sum.shape[1] - local_weights_sum_stragglers.shape[1],
                             0, local_weights_sum.shape[0] - local_weights_sum_stragglers.shape[0]),
                    value=0)
            elif len(local_weights_sum_stragglers.shape) == 2:
                padding = torch.nn.ConstantPad2d(
                    padding=(0, local_weights_sum.shape[1] - local_weights_sum_stragglers.shape[1],
                             0, local_weights_sum.shape[0] - local_weights_sum_stragglers.shape[0]),
                    value=0)
            else:
                padding = torch.nn.ConstantPad1d(
                    padding=(0, local_weights_sum.shape[0] - local_weights_sum_stragglers.shape[0]),
                    value=0)
            local_weights_sum += padding(local_weights_sum_stragglers)

        state_dict[key] = (local_weights_sum/args.num_users).to(state_dict[key].dtype)

    global_model.load_state_dict(state_dict)
    return


if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()

    N_iterations = args.monte_carlo_iterations
    mc_array = range(N_iterations)
    train_loss_mat = np.zeros((N_iterations, args.global_epochs))
    val_acc_mat = np.zeros((N_iterations, args.global_epochs))

    for n_itr in mc_array:

        textio, best_val_acc, path_best_model = utils.initializations(args)
        textio.cprint(str(args))

        # data
        train_data, test_loader = utils.data(args)
        input, output, train_data, val_loader = utils.data_split(train_data, len(test_loader.dataset), args)

        # model
        if args.model == 'mlp':
            global_model = models.FC2Layer(input, output)
        else:
            global_model = models.CNN2Layer(input, output, args.data)
        textio.cprint(str(summary(global_model)))
        global_model.to(args.device)

        train_creterion = torch.nn.CrossEntropyLoss(reduction='mean')
        test_creterion = torch.nn.CrossEntropyLoss(reduction='sum')

        # learning curve
        train_loss_list = []
        val_acc_list = []

        #  inference
        if args.eval:
            global_model.load_state_dict(torch.load(path_best_model))
            test_acc = utils.test(test_loader, global_model, test_creterion, args.device)
            textio.cprint(f'eval test_acc: {test_acc:.0f}%')
            gc.collect()
            sys.exit()

        # stragglers
        num_of_layers = global_model.state_dict().keys().__len__()
        if args.stragglers is not None:
            if args.sample_with_replacement:
                stragglers_idx = np.random.randint(low=0, high=args.num_users, size=round(args.stragglers_percent * args.num_users)) # randomly choose the stragglers
                stragglers_idx = list(set(stragglers_idx))
            else:
                stragglers_idx = random.sample(range(args.num_users), round(args.stragglers_percent * args.num_users)) # randomly choose the stragglers

            if args.model == 'mlp':
                mlp_inter_size_1, mlp_inter_size_2 = 16, 8
                intermidiate_global_model = models.FC2Layer(input, output, mlp_inter_size_1, mlp_inter_size_2).to(args.device)
            elif args.model == 'cnn2':
                cnn2_inter_size_1, cnn2_inter_size_2, cnn2_kernel_size = 3, 25, 3
                intermidiate_global_model = models.CNN2Layer(input, output, args.data, cnn2_kernel_size, cnn2_inter_size_1, cnn2_inter_size_2).to(args.device)
                data_size = intermidiate_global_model.data_size
            else:
                raise AssertionError('not a valid model')
        else:
            stragglers_idx = []
        non_stragglers_idx = np.setdiff1d(np.arange(args.num_users), stragglers_idx)

        local_models = hetroFL_federated_setup()

        for idx, global_epoch in tqdm(enumerate(range(0, args.global_epochs))):
            hetroFL_distribute_model()
            users_loss = []

            for user_idx in range(args.num_users):
                user_loss = []
                for local_epoch in range(0, args.local_epochs):
                    user = local_models[user_idx]
                    train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'], user['scheduler'],
                                                       train_creterion, args.device, args.local_iterations)
                    user_loss.append(train_loss)

                users_loss.append(mean(user_loss))

            train_loss = mean(users_loss)
            hetroFL_FedAvg()

            val_acc = utils.test(val_loader, global_model, test_creterion, args.device)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)

            train_loss_mat[n_itr, idx] = train_loss
            val_acc_mat[n_itr, idx] = val_acc
            gc.collect()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(global_model.state_dict(), path_best_model)

            test_acc = utils.test(test_loader, global_model, test_creterion, args.device)
            textio.cprint(f'epoch: {global_epoch} | train_loss: {train_loss:.2f} | '
                          f'val_acc: {val_acc:.0f}% | test_acc: {test_acc:.0f}%')

    mean_loss = np.mean(train_loss_mat, axis=0)
    mean_val_acc = np.mean(val_acc_mat, axis=0)
    np.save(f'checkpoints/{args.exp_name}/train_loss_list.npy', mean_loss)
    np.save(f'checkpoints/{args.exp_name}/val_acc_list.npy', mean_val_acc)
    np.save(f'checkpoints/{args.exp_name}/args.npy', args)

    elapsed_min = (time.time() - start_time) / 60
    textio.cprint(f'total execution time: {elapsed_min:.0f} min')

