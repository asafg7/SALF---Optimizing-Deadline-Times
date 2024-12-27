import numpy as np
import matplotlib.pyplot as plt

# MLP
trivial_allocation = np.load("checkpoints/27_12_24__mlp_inverse_uniform_deadlines/val_acc_list.npy")
optimal_allocation1 = np.load("checkpoints/27_12_24__mlp_inverse_optimal_deadlines/val_acc_list.npy")

iteration_time_trivial = np.load("checkpoints/27_12_24__mlp_inverse_uniform_deadlines/iteration_times.npy")
iteration_time_optimal1 = np.load("checkpoints/27_12_24__mlp_inverse_optimal_deadlines/iteration_times.npy")

time_trivial = np.cumsum(iteration_time_trivial)
time_optimal1 = np.cumsum(iteration_time_optimal1)

fig, ax = plt.subplots()
ax.plot(range(len(time_trivial)), iteration_time_trivial, label='Uniform')
ax.plot(range(len(time_optimal1)), iteration_time_optimal1, label='lr = 0.1')

ax.set(xlabel='iteration', ylabel='deadline time',
       title='Uniform vs Optimal Deadline Time Allocation - MLP')
ax.grid()
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(time_trivial, trivial_allocation, label='Uniform')
ax.plot(time_optimal1, optimal_allocation1, label='lr = 0.1')

ax.set(xlabel='Training time', ylabel='Validation Accuracy',
       title='Uniform vs Optimal Deadline Time Allocation - MLP')
ax.grid()
plt.legend()
plt.show()


# CNN
trivial_allocation = np.load("checkpoints/cnn_uniform_deadlines_decay_lr/val_acc_list.npy")
optimal_allocation = np.load("checkpoints/cnn_optimal_deadlines_decay_lr/val_acc_list.npy")

iteration_time_trivial = np.load("checkpoints/cnn_uniform_deadlines_decay_lr/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/cnn_optimal_deadlines_decay_lr/iteration_times.npy")

time_trivial = np.cumsum(iteration_time_trivial)
time_optimal = np.cumsum(iteration_time_optimal)

fig, ax = plt.subplots()
ax.plot(range(len(time_trivial)), iteration_time_trivial, label='Uniform')
ax.plot(range(len(time_optimal)), iteration_time_optimal, label='Optimal')

ax.set(xlabel='iteration', ylabel='deadline time',
       title='Uniform vs Optimal Deadline Time Allocation - CNN')
ax.grid()
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(time_trivial, trivial_allocation, label='Uniform')
ax.plot(time_optimal, optimal_allocation, label='Optimal')

ax.set(xlabel='Training time', ylabel='Validation Accuracy',
       title='Uniform vs Optimal Deadline Time Allocation - CNN')
ax.grid()
plt.legend()
plt.show()

# CNN 5
trivial_allocation = np.load("checkpoints/cnn_uniform_deadlines_decay_lr_highexp/val_acc_list.npy")
optimal_allocation = np.load("checkpoints/cnn_optimal_deadlines_decay_lrhighexp/val_acc_list.npy")

iteration_time_trivial = np.load("checkpoints/cnn_uniform_deadlines_decay_lr_highexp/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/cnn_optimal_deadlines_decay_lrhighexp/iteration_times.npy")

time_trivial = np.cumsum(iteration_time_trivial)
time_optimal = np.cumsum(iteration_time_optimal)

fig, ax = plt.subplots()
ax.plot(range(len(time_trivial)), iteration_time_trivial, label='Uniform')
ax.plot(range(len(time_optimal)), iteration_time_optimal, label='Optimal')

ax.set(xlabel='iteration', ylabel='deadline time',
       title='Uniform vs Optimal Deadline Time Allocation - CNN')
ax.grid()
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(time_trivial, trivial_allocation, label='Uniform')
ax.plot(time_optimal, optimal_allocation, label='Optimal')

ax.set(xlabel='Training time', ylabel='Validation Accuracy',
       title='Uniform vs Optimal Deadline Time Allocation - CNN')
ax.grid()
plt.legend()
plt.show()
