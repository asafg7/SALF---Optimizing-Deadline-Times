import numpy as np
import matplotlib.pyplot as plt
import fedavg_times

# MLP
uniform_salf_path = 'mnist_mlp_uniform_allocation_fixed_lr'
optimal_salf_path = 'mnist_mlp_inverse_allocation_fixed_lr'
fedavg_path = 'mnist_mlp_fedavg_fixed_lr'
drop_path = 'mnist_mlp_drop_fixed_lr'

iteration_time_uniform = np.load("checkpoints/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(15, 6, 1, 150)
iteration_time_drop = np.load("checkpoints/" + drop_path + "/iteration_times.npy")

validation_acc_uniform = np.load("checkpoints/" + uniform_salf_path + "/val_acc_list.npy")
validation_acc_optimal = np.load("checkpoints/" + optimal_salf_path + "/val_acc_list.npy")
validation_acc_fedavg = np.load("checkpoints/" + fedavg_path + "/val_acc_list.npy")
validation_acc_drop = np.load("checkpoints/" + drop_path + "/val_acc_list.npy")

time_uniform = np.cumsum(iteration_time_uniform)
time_optimal = np.cumsum(iteration_time_optimal)
time_fedavg = np.cumsum(iteration_time_fedavg)
time_drop = np.cumsum(iteration_time_drop)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("MLP - Fixed Learning Rate")
ax[0].plot(range(len(time_uniform)), iteration_time_uniform, label='Uniform Allocation')
ax[0].plot(range(len(time_optimal)), iteration_time_optimal, label='Allocation by inverse LR')

ax[0].set(xlabel='iteration', ylabel='deadline time',
       title='Deadline Allocation')
ax[0].grid()
ax[0].legend()

ax[1].plot(range(len(time_uniform)), validation_acc_uniform, label='Uniform Allocation')
ax[1].plot(range(len(time_optimal)), validation_acc_optimal, label='Allocation by inverse LR')
ax[1].plot(range(len(time_fedavg)), validation_acc_fedavg, label='FedAvg')
ax[1].plot(range(len(time_drop)), validation_acc_drop, label='Drop')

ax[1].set(xlabel='iteration', ylabel='Validation Accuracy [%]',
       title='Validation Accuracy by Iteration')
ax[1].grid()
ax[1].legend()

ax[2].plot(time_uniform, validation_acc_uniform, label='Uniform Allocation', marker="o", markevery=20, markersize=6)
ax[2].plot(time_optimal, validation_acc_optimal, label='Allocation by inverse LR', marker="o", markevery=20, markersize=6)
ax[2].plot(time_fedavg, validation_acc_fedavg, label='FedAvg', marker="o", markevery=20, markersize=6)
ax[2].plot(time_drop, validation_acc_drop, label='Drop', marker="o", markevery=20, markersize=6)
ax[2].set_xlim([0, 500])

ax[2].set(xlabel='Training time', ylabel='Validation Accuracy [%]',
       title='Validation Accuracy by Time')
ax[2].grid()
ax[2].legend()
plt.show()

# CNN
uniform_salf_path = 'mnist_cnn2_uniform_allocation_fixed_lr_rho'
optimal_salf_path = 'mnist_cnn2_inverse_allocation_fixed_lr_rho'
fedavg_path = 'mnist_cnn2_fedavg_fixed_lr02'
drop_path = 'mnist_cnn2_drop_fixed_lr02'
hetro_path = 'mnist_cnn2_hetro_fixed_lr02'

iteration_time_uniform = np.load("checkpoints/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(15, 8, 1, 70)
iteration_time_drop = np.load("checkpoints/" + drop_path + "/iteration_times.npy")
iteration_time_hetro = np.load("checkpoints/" + uniform_salf_path + "/iteration_times.npy")

validation_acc_uniform = np.load("checkpoints/" + uniform_salf_path + "/val_acc_list.npy")
validation_acc_optimal = np.load("checkpoints/" + optimal_salf_path + "/val_acc_list.npy")
validation_acc_fedavg = np.load("checkpoints/" + fedavg_path + "/val_acc_list.npy")
validation_acc_drop = np.load("checkpoints/" + drop_path + "/val_acc_list.npy")
validation_acc_hetro = np.load("checkpoints/" + hetro_path + "/val_acc_list.npy")

time_uniform = np.cumsum(iteration_time_uniform)
time_optimal = np.cumsum(iteration_time_optimal)
time_fedavg = np.cumsum(iteration_time_fedavg)
time_drop = np.cumsum(iteration_time_drop)
time_hetro = np.cumsum(iteration_time_hetro)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("CNN - Fixed Learning Rate")
ax[0].plot(range(len(time_uniform)), iteration_time_uniform, label='Uniform Allocation')
ax[0].plot(range(len(time_optimal)), iteration_time_optimal, label='Allocation by inverse LR')

ax[0].set(xlabel='iteration', ylabel='deadline time',
       title='Deadline Allocation')
ax[0].grid()
ax[0].legend()

ax[1].plot(range(len(time_uniform)), validation_acc_uniform, label='Uniform Allocation')
ax[1].plot(range(len(time_optimal)), validation_acc_optimal, label='Allocation by inverse LR')
ax[1].plot(range(len(time_fedavg)), validation_acc_fedavg, label='FedAvg')
ax[1].plot(range(len(time_drop)), validation_acc_drop, label='Drop')
ax[1].plot(range(len(time_hetro)), validation_acc_hetro, label='HetroFL')

ax[1].set(xlabel='iteration', ylabel='Validation Accuracy [%]',
       title='Validation Accuracy by Iteration')
ax[1].grid()
ax[1].legend()

ax[2].plot(time_uniform, validation_acc_uniform, label='Uniform Allocation', marker="o", markevery=20, markersize=6)
ax[2].plot(time_optimal, validation_acc_optimal, label='Allocation by inverse LR', marker="o", markevery=20, markersize=6)
ax[2].plot(time_fedavg, validation_acc_fedavg, label='FedAvg', marker="o", markevery=20, markersize=6)
ax[2].plot(time_drop, validation_acc_drop, label='Drop', marker="o", markevery=20, markersize=6)
ax[2].plot(time_hetro, validation_acc_hetro, label='HetroFL', marker="o", markevery=20, markersize=6)

ax[2].set(xlabel='Training time [s]', ylabel='Validation Accuracy [%]',
       title='Validation Accuracy by Time')
ax[2].set_xlim([0, 300])
ax[2].grid()
ax[2].legend()
plt.show()

# CNN SQRT
uniform_salf_path = 'mnist_cnn2_uniform_allocation_sqrt_lr'
optimal_salf_path = 'mnist_cnn2_inverse_allocation_sqrt_lr'
fedavg_path = 'mnist_cnn2_fedavg_sqrt_lr'
drop_path = 'mnist_cnn2_drop_sqrt_lr'
hetro_path = 'mnist_cnn2_hetro_sqrt_lr'

iteration_time_uniform = np.load("checkpoints/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(15, 8, 1, 70)
iteration_time_drop = np.load("checkpoints/" + drop_path + "/iteration_times.npy")
iteration_time_hetro = np.load("checkpoints/" + uniform_salf_path + "/iteration_times.npy")

validation_acc_uniform = np.load("checkpoints/" + uniform_salf_path + "/val_acc_list.npy")
validation_acc_optimal = np.load("checkpoints/" + optimal_salf_path + "/val_acc_list.npy")
validation_acc_fedavg = np.load("checkpoints/" + fedavg_path + "/val_acc_list.npy")
validation_acc_drop = np.load("checkpoints/" + drop_path + "/val_acc_list.npy")
validation_acc_hetro = np.load("checkpoints/" + hetro_path + "/val_acc_list.npy")

time_uniform = np.cumsum(iteration_time_uniform)
time_optimal = np.cumsum(iteration_time_optimal)
time_fedavg = np.cumsum(iteration_time_fedavg)
time_drop = np.cumsum(iteration_time_drop)
time_hetro = np.cumsum(iteration_time_hetro)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("CNN - Sqrt Learning Rate")
ax[0].plot(range(len(time_uniform)), iteration_time_uniform, label='Uniform Allocation')
ax[0].plot(range(len(time_optimal)), iteration_time_optimal, label='Allocation by inverse LR')

ax[0].set(xlabel='iteration', ylabel='deadline time',
       title='Deadline Allocation')
ax[0].grid()
ax[0].legend()

ax[1].plot(range(len(time_uniform)), validation_acc_uniform, label='Uniform Allocation')
ax[1].plot(range(len(time_optimal)), validation_acc_optimal, label='Allocation by inverse LR')
ax[1].plot(range(len(time_fedavg)), validation_acc_fedavg, label='FedAvg')
ax[1].plot(range(len(time_drop)), validation_acc_drop, label='Drop')
ax[1].plot(range(len(time_hetro)), validation_acc_hetro, label='HetroFL')

ax[1].set(xlabel='iteration', ylabel='Validation Accuracy [%]',
       title='Validation Accuracy by Iteration')
ax[1].grid()
ax[1].legend()

ax[2].plot(time_uniform, validation_acc_uniform, label='Uniform Allocation', marker="o", markevery=20, markersize=6)
ax[2].plot(time_optimal, validation_acc_optimal, label='Allocation by inverse LR', marker="o", markevery=20, markersize=6)
ax[2].plot(time_fedavg, validation_acc_fedavg, label='FedAvg', marker="o", markevery=20, markersize=6)
ax[2].plot(time_drop, validation_acc_drop, label='Drop', marker="o", markevery=20, markersize=6)
ax[2].plot(time_hetro, validation_acc_hetro, label='HetroFL', marker="o", markevery=20, markersize=6)

ax[2].set(xlabel='Training time [s]', ylabel='Validation Accuracy [%]',
       title='Validation Accuracy by Time')
ax[2].set_xlim([0, 300])
ax[2].grid()
ax[2].legend()
plt.show()
