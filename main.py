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

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    textio, best_val_acc, path_best_model = utils.initializations(args)
    textio.cprint(str(args))

    # data
    train_data, test_loader = utils.data(args)
    input, output, train_data, val_loader = utils.data_split(train_data, len(test_loader.dataset), args)

    iteration_times = [5.41064731, 5.15053798, 4.95472401, 4.79550987, 4.65995312, 4.54133918, 4.43547337, 4.33961966,
                       4.25189944,
                       4.17095886, 4.0957838, 4.02557874, 3.95971898, 3.89770171, 3.83910947, 3.78360423, 3.73088964,
                       3.6807262,
                       3.63289828, 3.58721885, 3.54352307, 3.50166861, 3.46152374, 3.42297317, 3.38591401, 3.35024897,
                       3.31589316,
                       3.28276694, 3.2507989, 3.21992181, 3.19008009, 3.16121029, 3.13326626, 3.10619866, 3.07996393,
                       3.05451965, 3.02982817, 3.00585242, 2.98255968, 2.95992207, 2.93790484, 2.9164837, 2.89563247,
                       2.87532615,
                       2.85554381, 2.83626051, 2.81745968, 2.79912029, 2.78122049, 2.76375202, 2.74668942, 2.73002226,
                       2.71373344,
                       2.69781185, 2.68224106, 2.66700879, 2.65210605, 2.63751759, 2.62323477, 2.60924905, 2.59554489,
                       2.58211965, 2.56895676, 2.5560547, 2.54339875, 2.53098442, 2.51880645, 2.50685431, 2.49512092,
                       2.48360191,
                       2.47229035, 2.46117754, 2.45025881, 2.43953178, 2.4289852, 2.41861872, 2.4084251, 2.39839921,
                       2.38854284, 2.37884042, 2.36929399, 2.35989816, 2.35065297, 2.34154846, 2.33258576, 2.32375555,
                       2.31506018,
                       2.30649406, 2.29805273, 2.289735, 2.281539868, 2.273458672, 2.265490994, 2.25763946, 2.24989390,
                       2.24225569,
                       2.23471975, 2.2272895, 2.2199552946, 2.212719668]
    iteration_times = np.ones(100)*3


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

    local_models = utils.federated_setup(global_model, train_data, args)

    # stragglers
    num_of_layers = global_model.state_dict().keys().__len__()
    if args.stragglers is not None:
        stragglers_idx = random.sample(range(args.num_users), round(args.stragglers_percent * args.num_users))
    else:
        stragglers_idx = []

    for global_epoch in tqdm(range(0, args.global_epochs)):
        utils.distribute_model(local_models, global_model)
        users_loss = []
        for user_idx in range(args.num_users):
            if (args.stragglers == 'drop') & (user_idx in stragglers_idx):
                user_new_state_dict = copy.deepcopy(global_model).state_dict()
                user_new_state_dict.update({})
                local_models[user_idx]['model'].load_state_dict(user_new_state_dict)
                continue

            user_loss = []
            for local_epoch in range(0, args.local_epochs):
                user = local_models[user_idx]
                train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'],
                                                   train_creterion, args.device, args.local_iterations)
                user_loss.append(train_loss)

            if (args.stragglers == 'salf') & (user_idx in stragglers_idx):
                user_new_state_dict = copy.deepcopy(global_model).state_dict()
                if args.up_to_layer is not None:
                    up_to_layer = num_of_layers - args.up_to_layer  # last-to-first layers updated
                else:
                    up_to_layer = np.random.randint(1, num_of_layers + 1)  # random last-to-first layers updated

                user_updated_layers = OrderedDict(islice(reversed(user['model'].state_dict().items()), up_to_layer))
                user_new_state_dict.update(user_updated_layers)
                user['model'].load_state_dict(user_new_state_dict)

            if (args.stragglers == 'opt_salf') & (user_idx in stragglers_idx):
                user_new_state_dict = copy.deepcopy(global_model).state_dict()
                if args.up_to_layer is not None:
                    up_to_layer = num_of_layers - args.up_to_layer  # last-to-first layers updated
                else:

                    up_to_layer = np.minimum(np.random.poisson(iteration_times[global_epoch]) + 1, num_of_layers)
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
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)

        gc.collect()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(global_model.state_dict(), path_best_model)

        textio.cprint(f'epoch: {global_epoch} | train_loss: {train_loss:.2f} | val_acc: {val_acc:.0f}%')

    np.save(f'checkpoints/{args.exp_name}/train_loss_list.npy', train_loss_list)
    np.save(f'checkpoints/{args.exp_name}/val_acc_list.npy', val_acc_list)


    elapsed_min = (time.time() - start_time) / 60
    textio.cprint(f'total execution time: {elapsed_min:.0f} min')
