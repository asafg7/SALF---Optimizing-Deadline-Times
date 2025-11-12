import numpy as np


def sample_erlang_max(n, k, lmbda, num_samples):
    # Generate n Erlang random variables and compute their max for each sample
    samples = np.max(np.random.gamma(shape=k, scale=1 / lmbda, size=(num_samples, n)), axis=1)
    return samples


def find_fedavg_iteration_time(n, k, lmbda, num_samples):

    # Generate samples
    max_samples = sample_erlang_max(n, k, lmbda, num_samples)
    return max_samples