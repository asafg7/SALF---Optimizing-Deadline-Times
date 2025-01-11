import numpy as np
import matplotlib.pyplot as plt

dir_list_uniform = ["27_12_24__mlp_inverse_uniform_deadlines",
                    "27_12_24__mlp_fixed_uniform_deadlines",
                    "27_12_24_mlp_sqrt_uniform_deadlines",
                    "27_12_24__cnn2_inverse_uniform_deadlines_mean4",
                    "27_12_24__cnn2_fixed_uniform_deadlines",
                    "27_12_24__cnn2_sqrt_uniform_deadlines"]

dir_list_optimal = ["27_12_24__mlp_inverse_optimal_deadlines",
                    "27_12_24__mlp_fixed_optimal_deadlines",
                    "27_12_24_mlp_sqrt_optimal_deadlines",
                    "27_12_24__cnn2_inverse_optimal_deadlines_mean4",
                    "27_12_24__cnn2_fixed_optimal_deadlines",
                    "27_12_24__cnn2_sqrt_optimal_deadlines"]

titles = ["Inverse decaying step size - MLP",
          "Constant step size - MLP",
          "Inverse square root step size - MLP",
          "Inverse decaying step size - CNN",
          "Constant step size - CNN",
          "Inverse square root step size - CNN"]

for idx in range(len(dir_list_uniform)):
    # load data
    uniform_allocation = np.load("checkpoints/"+dir_list_uniform[idx]+"/val_acc_list.npy")
    optimal_allocation = np.load("checkpoints/"+dir_list_optimal[idx]+"/val_acc_list.npy")
    iteration_time_uniform = np.load("checkpoints/"+dir_list_uniform[idx]+"/iteration_times.npy")
    iteration_time_optimal = np.load("checkpoints/"+dir_list_optimal[idx]+"/iteration_times.npy")
    time_uniform = np.cumsum(iteration_time_uniform)
    time_optimal = np.cumsum(iteration_time_optimal)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(range(len(iteration_time_uniform)), iteration_time_uniform,
            label='Uniform Deadline Allocation')
    axes[0].plot(range(len(iteration_time_optimal)), iteration_time_optimal,
            label='Special Deadline Allocation')
    axes[0].set(xlabel='iteration', ylabel='Deadline Time [s]')
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(time_uniform, uniform_allocation,
            label='Uniform Deadline Allocation')
    axes[1].plot(time_optimal, optimal_allocation,
            label='Special Deadline Allocation')
    axes[1].set(xlabel='Training time [s]', ylabel='Validation Accuracy [%]')
    axes[1].grid()
    axes[1].legend()

    fig.suptitle(titles[idx])
    plt.show()
