import numpy as np
import matplotlib.pyplot as plt

trivial_allocation = np.load("checkpoints/mlp_uniform_deadlines/val_acc_list.npy")
optimal_allocation = np.load("checkpoints/mlp_optimal_deadlines/val_acc_list.npy")

iteration_time_trivial = np.load("checkpoints/mlp_uniform_deadlines/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/mlp_optimal_deadlines/iteration_times.npy")

time_trivial = np.cumsum(iteration_time_trivial)
time_optimal = np.cumsum(iteration_time_optimal)

fig, ax = plt.subplots()
ax.plot(range(100), iteration_time_trivial, label='Trivial')
ax.plot(range(100), iteration_time_optimal, label='Optimal')

ax.set(xlabel='iteration', ylabel='deadline time',
       title='Trivial vs Optimal Iteration Time Allocation')
ax.grid()
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(time_trivial, trivial_allocation, label='Trivial')
ax.plot(time_optimal, optimal_allocation, label='Optimal')

ax.set(xlabel='Training time', ylabel='Validation Accuracy',
       title='Trivial vs Optimal Iteration Time Allocation')
ax.grid()
plt.legend()
plt.show()
