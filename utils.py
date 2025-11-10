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


def dirichlet_split_non_iid(dataset, alpha, n_clients):
    try:
        labels = np.array(dataset.targets)
    except AttributeError:
        # Fallback for datasets without a .targets attribute
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    n_classes = len(np.unique(labels))

    # client_indices[i] will be the list of indices for client i
    client_indices = [[] for _ in range(n_clients)]

    # class_indices[j] will be the list of indices for class j
    class_indices = [np.where(labels == i)[0] for i in range(n_classes)]

    for k_indices in class_indices:
        # For each class, get the indices, shuffle them
        np.random.shuffle(k_indices)

        # Sample proportions for this class from Dirichlet(alpha)
        # proportions[i] = fraction of class k given to client i
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))

        # Correct for potential rounding errors by ensuring sum is exactly len(k_indices)
        proportions = (proportions * len(k_indices)).astype(int)
        proportions[-1] = len(k_indices) - np.sum(proportions[:-1])

        # Split the class indices based on the sampled proportions
        current_idx = 0
        for i in range(n_clients):
            client_indices[i].extend(k_indices[current_idx: current_idx + proportions[i]])
            current_idx += proportions[i]

    # Shuffle each client's final list of indices
    for i in range(n_clients):
        np.random.shuffle(client_indices[i])

    return client_indices

def federated_setup(global_model, train_data, args, train_batch_size):
    # create a dict of dict s (local users), i.e. {'1': {'data':..., 'model':..., 'opt':...}, ...}

    if args.dirichlet_alpha is not None:
        if args.num_samples is not None:
            raise ValueError("Cannot use --dirichlet_alpha and --num_samples together. "
                             "The Dirichlet split partitions all available data.")

        client_indices = dirichlet_split_non_iid(train_data, args.dirichlet_alpha, args.num_users)

    else:
        print("Generating IID split...")
        indexes = torch.randperm(len(train_data))
        user_data_len = math.floor(len(train_data) / args.num_users) if args.num_samples == None else args.num_samples
        client_indices = []
        for user_idx in range(args.num_users):
            start = user_idx * user_data_len
            end = (user_idx + 1) * user_data_len
            client_indices.append(indexes[start:end])


    local_models = {}
    if args.lr_decay == "fixed":
        lambda_func = lambda epoch: args.lr
    elif args.lr_decay == "inverse":
        lambda_func = lambda epoch: 1 / (1 + epoch)

    for user_idx in range(args.num_users):
        user_indices = client_indices[user_idx]

        if len(user_indices) == 0:
            print(f"Warning: Client {user_idx} has 0 samples. This can happen with extreme non-IID (low alpha).")
            user_subset = torch.utils.data.Subset(train_data, [])
        else:
            user_subset = torch.utils.data.Subset(train_data, user_indices)

        user = {'data': torch.utils.data.DataLoader(
            user_subset,
            batch_size=round(train_batch_size[user_idx]), shuffle=True),
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
        layer_to_stop_arr = [0, 1, 2, 3, 4, 5, 6]
    elif model == 'cnn2':
        num_layers = 8
        layer_to_stop_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8]
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
        layer_to_stop_arr = []
    return (num_layers, layer_to_stop_arr)

def get_samples_count(data):
    if data == 'mnist':
        N_samples = 60000
    elif data == 'cifar10':
        N_samples = 50000
    else:
        N_samples = 0
    return N_samples
