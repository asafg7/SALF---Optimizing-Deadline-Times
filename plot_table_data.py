import numpy as np
import matplotlib.pyplot as plt
import fedavg_times

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Sub Figure a - deadline allocation

path1 = 'article_vgg11_cifar_uniform_momentum'
path2 = 'table_article_vgg11_cifar_fulloptimization_momentum'

label1 = 'SALF'
label2 = 'ADEL-FL'

iteration_time_1 = np.load("checkpoints/" + path1 + "/iteration_times.npy")
iteration_time_2 = np.load("checkpoints/" + path2 + "/iteration_times.npy")

ax[0].plot(range(len(iteration_time_1)), iteration_time_1, label='Constant Deadline', linestyle='-')
ax[0].plot(range(len(iteration_time_2)), iteration_time_2, label=label2, linestyle='--')

ax[0].set_xlabel('Round \n (a) deadline allocation for CIFAR10 VGG11', fontsize=15)
ax[0].set_ylabel('Deadline Time [s]', fontsize=15)
ax[0].grid()
ax[0].legend()

# Sub Figure b - VGG11 CIFAR10

path1 = 'article_vgg11_cifar_uniform_momentum'
path2 = 'table_article_vgg11_cifar_fulloptimization_momentum'
path3 = 'article_vgg11_cifar_fedavg_momentum'
path4 = 'article_vgg11_cifar_drop_momentum'

label1 = 'SALF'
label2 = 'ADEL-FL'
label3 = 'Wait Stragglers'
label4 = 'FedAvg'

iteration_time_1 = np.load("checkpoints/" + path1 + "/iteration_times.npy")
iteration_time_2 = np.load("checkpoints/" + path2 + "/iteration_times.npy")
iteration_time_3 = fedavg_times.find_fedavg_iteration_time(20, 11, 1, 300)
iteration_time_4 = np.load("checkpoints/" + path4 + "/iteration_times.npy")

validation_1 = np.load("checkpoints/" + path1 + "/val_acc_list.npy")
validation_2 = np.load("checkpoints/" + path2 + "/val_acc_list.npy")
validation_3 = np.load("checkpoints/" + path3 + "/val_acc_list.npy")
validation_4 = np.load("checkpoints/" + path4 + "/val_acc_list.npy")

time_1 = np.cumsum(iteration_time_1)
time_2 = np.cumsum(iteration_time_2)
time_3 = np.cumsum(iteration_time_3)
time_4 = np.cumsum(iteration_time_4)

ax[1].plot(time_1, validation_1, label=label1, marker="o", markevery=20, markersize=6, linestyle='-')
ax[1].plot(time_2, validation_2, label=label2, marker="o", markevery=20, markersize=6, linestyle='--')
ax[1].plot(time_3, validation_3, label=label3, marker="o", markevery=20, markersize=6, linestyle='-.')
ax[1].plot(time_4, validation_4, label=label4, marker="o", markevery=20, markersize=6, linestyle=':')

ax[1].set_xlabel('Training Time [s] \n (b) CIFAR10 VGG11 Convergence', fontsize=15)
ax[1].set_ylabel('Validation Accuracy', fontsize=15)
ax[1].set(xlim=[0, 2700])
ax[1].grid()
ax[1].legend()

# Sub Figure c - VGG13 CIFAR10

path1 = 'fix_article_vgg13_cifar_uniform_momentum'
path2 = 'fix_article_vgg13_cifar_fulloptimization_momentum'
path3 = 'fix_article_vgg13_cifar_fedavg_momentum'
path4 = 'fix_article_vgg13_cifar_drop_momentum'

label1 = 'SALF'
label2 = 'ADEL-FL'
label3 = 'Wait Stragglers'
label4 = 'FedAvg'

iteration_time_1 = np.load("checkpoints/" + path1 + "/iteration_times.npy")
iteration_time_2 = np.load("checkpoints/" + path2 + "/iteration_times.npy")
iteration_time_3 = fedavg_times.find_fedavg_iteration_time(20, 13, 1, 300)
iteration_time_4 = np.load("checkpoints/" + path4 + "/iteration_times.npy")

validation_1 = np.load("checkpoints/" + path1 + "/val_acc_list.npy")
validation_2 = np.load("checkpoints/" + path2 + "/val_acc_list.npy")
validation_3 = np.load("checkpoints/" + path3 + "/val_acc_list.npy")
validation_4 = np.load("checkpoints/" + path4 + "/val_acc_list.npy")

time_1 = np.cumsum(iteration_time_1)
time_2 = np.cumsum(iteration_time_2)
time_3 = np.cumsum(iteration_time_3)
time_4 = np.cumsum(iteration_time_4)

ax[2].plot(time_1, validation_1, label=label1, marker="o", markevery=20, markersize=6, linestyle='-')
ax[2].plot(time_2, validation_2, label=label2, marker="o", markevery=20, markersize=6, linestyle='--')
ax[2].plot(time_3, validation_3, label=label3, marker="o", markevery=20, markersize=6, linestyle='-.')
ax[2].plot(time_4, validation_4, label=label4, marker="o", markevery=20, markersize=6, linestyle=':')

ax[2].set_xlabel('Training Time [s] \n (c) CIFAR10 VGG13 convergence', fontsize=15)
ax[2].set_ylabel('Validation Accuracy', fontsize=15)
ax[2].set(xlim=[0, 3300])
ax[2].grid()
ax[2].legend()

# end
plt.tight_layout()
plt.savefig('fig_cifar.pdf', bbox_inches='tight')
plt.show()
