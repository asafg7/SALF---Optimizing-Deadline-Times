import numpy as np
import matplotlib.pyplot as plt

# MLP
trivial_allocation = np.load("checkpoints/mlp_optimal_deadlines_uniform/val_acc_list.npy")
optimal_allocation1 = np.load("checkpoints/mlp_optimal_deadlines_fixedlr/val_acc_list.npy")
optimal_allocation2 = np.load("checkpoints/mlp_optimal_deadlines_lindecaylr/val_acc_list.npy")
optimal_allocation3 = np.load("checkpoints/mlp_optimal_deadlines_changed_lr/val_acc_list.npy")

iteration_time_trivial = np.load("checkpoints/mlp_optimal_deadlines_uniform/iteration_times.npy")
iteration_time_optimal1 = np.load("checkpoints/mlp_optimal_deadlines_fixedlr/iteration_times.npy")
iteration_time_optimal2 = np.load("checkpoints/mlp_optimal_deadlines_lindecaylr/iteration_times.npy")
iteration_time_optimal3 = np.load("checkpoints/mlp_optimal_deadlines_changed_lr/iteration_times.npy")

time_trivial = np.cumsum(iteration_time_trivial)
time_optimal1 = np.cumsum(iteration_time_optimal1)
time_optimal2 = np.cumsum(iteration_time_optimal2)
time_optimal3 = np.cumsum(iteration_time_optimal3)

fig, ax = plt.subplots()
ax.plot(range(len(time_trivial)), iteration_time_trivial, label='Uniform')
ax.plot(range(len(time_optimal1)), iteration_time_optimal1, label='lr = 0.1')
ax.plot(range(len(time_optimal2)), iteration_time_optimal2, label=r"$lr = \frac{2}{0.5*t + 1}$")
ax.plot(range(len(time_optimal3)), iteration_time_optimal3, label=r"$get \; T_{t}^{d} \; by \;  \frac {1}{1+t}, \;train \; network \; by \; \frac{2}{0.5*t + 1}$")

ax.set(xlabel='iteration', ylabel='deadline time',
       title='Uniform vs Optimal Deadline Time Allocation - MLP')
ax.grid()
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(time_trivial, trivial_allocation, label='Uniform')
ax.plot(time_optimal1, optimal_allocation1, label='lr = 0.1')
ax.plot(time_optimal2, optimal_allocation2, label=r"$lr = \frac{2}{0.5*t + 1}$")
ax.plot(time_optimal3, optimal_allocation3, label=r"$get \; T_{t}^{d} \; by \;  \frac {1}{1+t}, \;train \; network \; by \; \frac{2}{0.5*t + 1}$")

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
