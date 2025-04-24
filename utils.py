import math
import os
import random
import time

import torch
import copy
import numpy as np
from torch import optim
from statistics import mean
from torchvision import datasets, transforms


def FedAvg(local_models, global_model):
    state_dict = global_model.state_dict()
    for key in state_dict.keys():
        local_weights_sum = torch.zeros_like(state_dict[key])
        count_updating_users = 0
        for user_idx in range(0, len(local_models)):
            if key in local_models[user_idx]['model'].state_dict():
                local_weights_sum += local_models[user_idx]['model'].state_dict()[key]
                count_updating_users += 1
        if count_updating_users != 0:
            state_dict[key] = (local_weights_sum / len(local_models)).to(state_dict[key].dtype)

    global_model.load_state_dict(state_dict)
    return


def federated_setup(global_model, train_data, args):
    # create a dict of dict s (local users), i.e. {'1': {'data':..., 'model':..., 'opt':...}, ...}
    indexes = torch.randperm(len(train_data))
    user_data_len = math.floor(len(train_data) / args.num_users) if args.num_samples == None else args.num_samples
    local_models = {}
    if args.lr_decay == "fixed":
        lambda_func = lambda epoch: args.lr
    elif args.lr_decay == "inverse":
        lambda_func = lambda epoch: 1 / (args.rho_c * (epoch + np.max((8 * args.rho_s / args.rho_c, 1)) - 1))
    elif args.lr_decay == "sqrt":
        lambda_func = lambda epoch: 1 / (args.rho_c * (np.sqrt(epoch) + np.max((8 * args.rho_s / args.rho_c, 1)) - 1))
    for user_idx in range(args.num_users):
        user = {'data': torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data,
                                    indexes[user_idx * user_data_len:(user_idx + 1) * user_data_len]),
            batch_size=args.train_batch_size, shuffle=True),
            'model': copy.deepcopy(global_model)}
        user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay) if args.optimizer == 'sgd' \
            else optim.Adam(user['model'].parameters(), lr=args.lr)
        user['scheduler'] = optim.lr_scheduler.LambdaLR(user['opt'], lr_lambda=lambda_func)
        local_models[user_idx] = user
    return local_models


def distribute_model(local_models, global_model):
    for user_idx in range(len(local_models)):
        local_models[user_idx]['model'].load_state_dict(global_model.state_dict())


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        # self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def initializations(args):
    #  reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #  documentation
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')

    best_val_acc = -np.inf
    path_best_model = 'checkpoints/' + args.exp_name + '/model.best.t7'

    return textio, best_val_acc, path_best_model


def data(args):
    if args.data == 'mnist':
        train_data = datasets.MNIST('./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                    ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    else:
        train_data = datasets.CIFAR10('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                      ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    return train_data, test_loader


def data_split(data, amount, args):
    # split train, validation
    train_data, val_data = torch.utils.data.random_split(data, [len(data) - amount, amount])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False)

    # input, output sizes
    in_channels, dim1, dim2 = data[0][0].shape  # images are dim1 x dim2 pixels
    input = dim1 * dim2 if args.model == 'mlp' or args.model == 'linear' else in_channels
    output = len(data.classes)  # number of classes

    return input, output, train_data, val_loader


def train_one_epoch(train_loader, model, optimizer, scheduler,
                    creterion, device, iterations):
    model.train()
    losses = []
    if iterations is not None:
        local_iteration = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # send to device
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = creterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        start = time.time()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if iterations is not None:
            local_iteration += 1
            if local_iteration == iterations:
                break
    return mean(losses)


def test(test_loader, model, creterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)  # send to device

        output = model(data)
        test_loss += creterion(output, label).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def get_layers(model):
    if model == 'mlp':
        num_layers = 6
    elif model == 'cnn2':
        num_layers = 8
    elif model == "VGG11":
        num_layers = 11
        layer_to_stop_arr = [0, 2, 2, 2, 9, 16, 23, 30, 37, 44, 51, 58]
    elif model == 'VGG13':
        num_layers = 13
        layer_to_stop_arr = [0, 2, 2, 2, 9, 16, 23, 30, 37, 44, 51, 58, 65, 72]
    elif model == 'VGG16':
        num_layers = 16
        layer_to_stop_arr = [0, 2, 2, 2, 9, 16, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86]
    elif model == 'VGG19':
        num_layers = 19
        layer_to_stop_arr = [0, 2, 2, 2, 9, 16, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93, 100]
    else:
        num_layers = 0
        layer_to_stop = []
    return (num_layers, layer_to_stop_arr)

def get_samples_count(data):
    if data == 'mnist':
        N_samples = 60000
    elif data == 'cifar10':
        N_samples = 50000
    else:
        N_samples = 0
    return N_samples
