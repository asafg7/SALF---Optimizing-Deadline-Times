import numpy as np
import matplotlib.pyplot as plt
import fedavg_times

fig, ax = plt.subplots(2, 2, figsize=(15, 15))

# MLP
uniform_salf_path = 'salf_mlp_mnist'
optimal_salf_path = 'adel_fl_mlp_mnist'
fedavg_path = 'fedavg_mlp_mnist'
drop_path = 'drop_mlp_mnist'

iteration_time_uniform = np.load("checkpoints/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(20, 8, 1, 100)
iteration_time_drop = np.load("checkpoints/" + drop_path + "/iteration_times.npy")

validation_acc_uniform = np.load("checkpoints/" + uniform_salf_path + "/val_acc_list.npy")
validation_acc_optimal = np.load("checkpoints/" + optimal_salf_path + "/val_acc_list.npy")
validation_acc_fedavg = np.load("checkpoints/" + fedavg_path + "/val_acc_list.npy")
validation_acc_drop = np.load("checkpoints/" + drop_path + "/val_acc_list.npy")

time_uniform = np.cumsum(iteration_time_uniform)
time_optimal = np.cumsum(iteration_time_optimal)
time_fedavg = np.cumsum(iteration_time_fedavg)
time_drop = np.cumsum(iteration_time_drop)

ax[0, 0].plot(range(len(time_uniform)), iteration_time_uniform, label='Uniform Allocation')
ax[0, 0].plot(range(len(time_optimal)), iteration_time_optimal, label='Allocation by inverse LR')

ax[0, 0].set(xlabel='iteration', ylabel='deadline time',
       title='Deadline Allocation - MLP')
ax[0, 0].grid()
ax[0, 0].legend()

ax[0, 1].plot(time_uniform, validation_acc_uniform, label='SALF', marker="o", markevery=20, markersize=6)
ax[0, 1].plot(time_optimal, validation_acc_optimal, label='ADEL-FL', marker="o", markevery=20, markersize=6)
ax[0, 1].plot(time_fedavg, validation_acc_fedavg, label='FedAvg', marker="o", markevery=20, markersize=6)
ax[0, 1].plot(time_drop, validation_acc_drop, label='Drop', marker="o", markevery=20, markersize=6)
ax[0, 1].set_xlim([0, 400])

ax[0, 1].set(xlabel='Training time', ylabel='Validation Accuracy [%]',
       title='Validation Accuracy by Time - MLP')
ax[0, 1].grid()
ax[0, 1].legend()

# CNN
uniform_salf_path = 'salf_cnn_mnist'
optimal_salf_path = 'adel_fl_cnn_mnist'
fedavg_path = 'fedavg_cnn_mnist'
drop_path = 'drop_cnn_mnist'

iteration_time_uniform = np.load("checkpoints/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(20, 8, 1, 50)
iteration_time_drop = np.load("checkpoints/" + drop_path + "/iteration_times.npy")

validation_acc_uniform = np.load("checkpoints/" + uniform_salf_path + "/val_acc_list.npy")
validation_acc_optimal = np.load("checkpoints/" + optimal_salf_path + "/val_acc_list.npy")
validation_acc_fedavg = np.load("checkpoints/" + fedavg_path + "/val_acc_list.npy")
validation_acc_drop = np.load("checkpoints/" + drop_path + "/val_acc_list.npy")

time_uniform = np.cumsum(iteration_time_uniform)
time_optimal = np.cumsum(iteration_time_optimal)
time_fedavg = np.cumsum(iteration_time_fedavg)
time_drop = np.cumsum(iteration_time_drop)

ax[1, 0].plot(range(len(time_uniform)), iteration_time_uniform, label='Uniform Allocation')
ax[1, 0].plot(range(len(time_optimal)), iteration_time_optimal, label='Allocation by inverse LR')

ax[1, 0].set(xlabel='iteration', ylabel='deadline time',
       title='Deadline Allocation - CNN')
ax[1, 0].grid()
ax[1, 0].legend()

ax[1, 1].plot(time_uniform, validation_acc_uniform, label='SALF', marker="o", markevery=20, markersize=6)
ax[1, 1].plot(time_optimal, validation_acc_optimal, label='ADEL-FL', marker="o", markevery=20, markersize=6)
ax[1, 1].plot(time_fedavg, validation_acc_fedavg, label='FedAvg', marker="o", markevery=20, markersize=6)
ax[1, 1].plot(time_drop, validation_acc_drop, label='Drop', marker="o", markevery=20, markersize=6)

ax[1, 1].set(xlabel='Training time [s]', ylabel='Validation Accuracy [%]',
       title='Validation Accuracy by Time - CNN')
ax[1, 1].set_xlim([0, 200])
ax[1, 1].grid()
ax[1, 1].legend()
plt.show()
