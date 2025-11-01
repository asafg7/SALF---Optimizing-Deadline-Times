import argparse

def none_or_str(value):
    if value == 'None':
        return None
    return value

def float_or_str(value):
    if type(value) == str:
        return float(value)
    return value

def int_or_str(value):
    if type(value) == str:
        if value == 'None':
            return None
        return int(value)
    return value

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='exp',
                        help="the name of the current experiment")
    parser.add_argument('--stragglers', type=none_or_str, default='bcd',
                        choices=['drop', None, 'adel-fl'],
                        help="whether the FL is stragglers aware")
    parser.add_argument('--stragglers_percent', type=float_or_str, default=1,
                        help="the percent of percent out of the edge users")
    parser.add_argument('--up_to_layer', type=int_or_str, default=None,
                        help="if 'None' - choose randomly, else - update until (num_layers - up_to_layer)"
                             "example: up_to_layer=1 results with an update up to one before the first layer")
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist', 'cifar10'],
                        help="dataset to use (mnist or cifar)")
    parser.add_argument('--model', type=str, default='cnn2',
                        choices=['mlp', 'cnn2', 'VGG11', 'VGG13', 'VGG16', 'VGG19'],
                        help="model arcitecture to be used")
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help="device to use (gpu or cpu)")

    parser.add_argument('--num_samples', type=int, default=None,
                        help="number of samples per user; if 'None' - uniformly distribute all data among all users)")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users participating in the federated learning")
    parser.add_argument('--train_batch_size', type=int, default=128,
                        help="trainset batch size")
    parser.add_argument('--local_iterations', type=int, default=1,
                        help="number of local iterations instead of local epoch")
    parser.add_argument('--norm_mean', type=float, default=0.5,
                        help="normalize the data to norm_mean")
    parser.add_argument('--norm_std', type=float, default=0.5,
                        help="normalize the data to norm_std")
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help="testset batch size")
    parser.add_argument('--local_epochs', type=int, default=1,
                        help="number of local epochs")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help="optimizer to use (sgd or adam)")
    parser.add_argument('--seed', type=float, default=1234,
                        help="manual seed for reproducibility")
    parser.add_argument('--eval', action='store_true',
                        help="weather to perform inference of training")
    parser.add_argument('--monte_carlo_iterations', type=int, default=1,
                        help="number of iterations for model training")
    parser.add_argument('--sample_with_replacement', type=bool, default=True,
                        help="sample_with_replacement")

    parser.add_argument('--deadline_times', type=str, default='inverse',
                        choices=['uniform', 'inverse', 'fixed'],
                        help="perform optimization according to learning rate")
    parser.add_argument('--lr_decay', type=str, default="inverse",
                        choices=['inverse', 'fixed'],
                        help="learning rate decay")
    parser.add_argument('--batchsize_optimization', type=bool, default=True,
                        help="std of the user sgd")

    parser.add_argument('--global_epochs', type=int, default=20,
                        help="number of global epochs")
    parser.add_argument('--t_max', type=int, default=120,
                        help="maximal training time for adel-fl")
    parser.add_argument('--t_min', type=float, default=3.4,
                        help="iteration minimum time")
    parser.add_argument('--weight_decay', type=float, default=0,
                        help="l2 regularization")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="momentum")

    parser.add_argument('--g', type=float, default=0.1,
                        help="gradient bound")
    parser.add_argument('--rho_s', type=float, default=1,
                        help="smoothness constant")
    parser.add_argument('--rho_c', type=float, default=3e-1,
                        help="strong convexity constant")
    parser.add_argument('--gamma', type=float, default=1,
                        help="heterogeneity gap")
    parser.add_argument('--lr', type=float, default=0.5,
                        help="Learning rate for fixed rate")
    parser.add_argument('--mean_std', type=float, default=5,
                        help="std of the user sgd")
    parser.add_argument('--alpha', type=float, default=1,
                        help="sgd variance weight")
    parser.add_argument('--maxR', type=float, default=8192,
                        help="sgd variance weight")

    args = parser.parse_args()
    return args
