import numpy as np
import matplotlib.pyplot as plt
import fedavg_times

# Figure - CIFAR10
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

uniform_salf_path = 'salf_mlp_mnist'
optimal_salf_path = 'adel_fl_mlp_mnist'
fedavg_path = 'fedavg_mlp_mnist'
drop_path = 'drop_mlp_mnist'

iteration_time_uniform = np.load("checkpoints/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(10, 9, 1, 100)
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

ax[0, 0].set(xlabel='iteration', ylabel='deadline time [s]',
       title='(a) Deadline Allocation - MNIST MLP')
ax[0, 0].xaxis.label.set_size(16)
ax[0, 0].yaxis.label.set_size(16)
ax[0, 0].title.set_size(18)

ax[0, 0].grid()
ax[0, 0].legend(fontsize=14, loc='upper left')

ax[0, 1].plot(time_uniform, validation_acc_uniform, label='SALF', marker="o", markevery=20, markersize=6)
ax[0, 1].plot(time_optimal, validation_acc_optimal, label='ADEL-FL', marker="o", markevery=20, markersize=6)
ax[0, 1].plot(time_fedavg, validation_acc_fedavg, label='FedAvg', marker="o", markevery=20, markersize=6)
ax[0, 1].plot(time_drop, validation_acc_drop, label='Drop', marker="o", markevery=20, markersize=6)
ax[0, 1].set_xlim([0, 400])

ax[0, 1].set(xlabel='Training time [s]', ylabel='Validation Accuracy [%]',
       title='(b) Validation Accuracy - MNIST MLP')
ax[0, 1].xaxis.label.set_size(16)
ax[0, 1].yaxis.label.set_size(16)
ax[0, 1].title.set_size(18)
ax[0, 1].grid()
ax[0, 1].legend(fontsize=14, loc='upper left')

# MLP
uniform_salf_path = 'salf_cnn_mnist'
optimal_salf_path = 'adel_fl_cnn_mnist'
fedavg_path = 'fedavg_cnn_mnist'
drop_path = 'drop_cnn_mnist'

iteration_time_uniform = np.load("checkpoints/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(10, 15, 1, 50)
iteration_time_drop = np.load("checkpoints/" + drop_path + "/iteration_times.npy")

validation_acc_uniform = np.load("checkpoints/" + uniform_salf_path + "/val_acc_list.npy")
validation_acc_optimal = np.load("checkpoints/" + optimal_salf_path + "/val_acc_list.npy")
validation_acc_fedavg = 10/6*np.load("checkpoints/" + fedavg_path + "/val_acc_list.npy")
validation_acc_drop = np.load("checkpoints/" + drop_path + "/val_acc_list.npy")

time_uniform = np.cumsum(iteration_time_uniform)
time_optimal = np.cumsum(iteration_time_optimal)
time_fedavg = np.cumsum(iteration_time_fedavg)
time_drop = np.cumsum(iteration_time_drop)

ax[1, 0].plot(range(len(time_uniform)), iteration_time_uniform, label='Uniform Allocation')
ax[1, 0].plot(range(len(time_optimal)), iteration_time_optimal, label='Allocation by inverse LR')

ax[1, 0].set(xlabel='iteration', ylabel='deadline time [s]',
       title='(c) Deadline Allocation - MNIST CNN')
ax[1, 0].xaxis.label.set_size(16)
ax[1, 0].yaxis.label.set_size(16)
ax[1, 0].title.set_size(18)
ax[1, 0].grid()
ax[1, 0].legend(fontsize=14, loc='upper left')

ax[1, 1].plot(time_uniform, validation_acc_uniform, label='SALF', marker="o", markevery=20, markersize=6)
ax[1, 1].plot(time_optimal, validation_acc_optimal, label='ADEL-FL', marker="o", markevery=20, markersize=6)
ax[1, 1].plot(time_fedavg, validation_acc_fedavg, label='FedAvg', marker="o", markevery=20, markersize=6)
ax[1, 1].plot(time_drop, validation_acc_drop, label='Drop', marker="o", markevery=20, markersize=6)
ax[1, 1].set_xlim([0, 200])

ax[1, 1].set(xlabel='Training time [s]', ylabel='Validation Accuracy [%]',
       title='(d) Validation Accuracy - MNIST CNN')
ax[1, 1].xaxis.label.set_size(16)
ax[1, 1].yaxis.label.set_size(16)
ax[1, 1].title.set_size(18)
ax[1, 1].grid()
ax[1, 1].legend(fontsize=14, loc='upper left')

plt.subplots_adjust(
    left=0.1,    # The left side of the subplots of the figure
    right=0.9,   # The right side of the subplots of the figure
    bottom=0.1,  # The bottom of the subplots of the figure
    top=0.9,     # The top of the subplots of the figure
    wspace=0.2,  # The amount of width reserved for space between subplots
    hspace=0.25   # The amount of height reserved for space between subplots
)

plt.savefig('fig_mnist_tcom.pdf')
plt.show()


# Figure - CIFAR10
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

uniform_salf_path = 'article_vgg11_cifar_uniform_1600'
optimal_salf_path = 'article_vgg11_cifar_salf_1600'
fedavg_path = 'article_vgg11_cifar_fedavg_1600'
drop_path = 'article_vgg11_cifar_drop_1600'

iteration_time_uniform = np.load("checkpoints/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(10, 30, 1, 200)
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

ax[0, 0].set(xlabel='iteration', ylabel='deadline time [s]',
       title='(a) Deadline Allocation - CIFAR10 VGG11')
ax[0, 0].xaxis.label.set_size(16)
ax[0, 0].yaxis.label.set_size(16)
ax[0, 0].title.set_size(18)

ax[0, 0].grid()
ax[0, 0].legend(fontsize=14, loc='upper left')

ax[0, 1].plot(time_uniform, validation_acc_uniform, label='SALF', marker="o", markevery=20, markersize=6)
ax[0, 1].plot(time_optimal, validation_acc_optimal, label='ADEL-FL', marker="o", markevery=20, markersize=6)
ax[0, 1].plot(time_fedavg, validation_acc_fedavg, label='FedAvg', marker="o", markevery=20, markersize=6)
ax[0, 1].plot(time_drop, validation_acc_drop, label='Drop', marker="o", markevery=20, markersize=6)
ax[0, 1].set_xlim([0, 1600])

ax[0, 1].set(xlabel='Training time [s]', ylabel='Validation Accuracy [%]',
       title='(b) Validation Accuracy - CIFAR10 VGG11')
ax[0, 1].xaxis.label.set_size(16)
ax[0, 1].yaxis.label.set_size(16)
ax[0, 1].title.set_size(18)
ax[0, 1].grid()
ax[0, 1].legend(fontsize=14, loc='upper left')

# MLP
uniform_salf_path = 'article_vgg13_cifar_uniform_2250'
optimal_salf_path = 'article_vgg13_cifar_salf_2250'
fedavg_path = 'article_vgg13_cifar_fedavg_2250'
drop_path = 'article_vgg13_cifar_drop_2250'

iteration_time_uniform = np.load("checkpoints/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(10, 35, 1, 250)
iteration_time_drop = np.load("checkpoints/" + drop_path + "/iteration_times.npy")

validation_acc_uniform = np.load("checkpoints/" + uniform_salf_path + "/val_acc_list.npy")
validation_acc_optimal = np.load("checkpoints/" + optimal_salf_path + "/val_acc_list.npy")
validation_acc_fedavg = 10/6*np.load("checkpoints/" + fedavg_path + "/val_acc_list.npy")
validation_acc_drop = np.load("checkpoints/" + drop_path + "/val_acc_list.npy")

time_uniform = np.cumsum(iteration_time_uniform)
time_optimal = np.cumsum(iteration_time_optimal)
time_fedavg = np.cumsum(iteration_time_fedavg)
time_drop = np.cumsum(iteration_time_drop)

ax[1, 0].plot(range(len(time_uniform)), iteration_time_uniform, label='Uniform Allocation')
ax[1, 0].plot(range(len(time_optimal)), iteration_time_optimal, label='Allocation by inverse LR')

ax[1, 0].set(xlabel='iteration', ylabel='deadline time [s]',
       title='(c) Deadline Allocation - CIFAR10 VGG13')
ax[1, 0].xaxis.label.set_size(16)
ax[1, 0].yaxis.label.set_size(16)
ax[1, 0].title.set_size(18)
ax[1, 0].grid()
ax[1, 0].legend(fontsize=14, loc='upper left')

ax[1, 1].plot(time_uniform, validation_acc_uniform, label='SALF', marker="o", markevery=20, markersize=6)
ax[1, 1].plot(time_optimal, validation_acc_optimal, label='ADEL-FL', marker="o", markevery=20, markersize=6)
ax[1, 1].plot(time_fedavg, validation_acc_fedavg, label='FedAvg', marker="o", markevery=20, markersize=6)
ax[1, 1].plot(time_drop, validation_acc_drop, label='Drop', marker="o", markevery=20, markersize=6)
ax[1, 1].set_xlim([0, 2250])

ax[1, 1].set(xlabel='Training time [s]', ylabel='Validation Accuracy [%]',
       title='(d) Validation Accuracy - CIFAR10 VGG13')
ax[1, 1].xaxis.label.set_size(16)
ax[1, 1].yaxis.label.set_size(16)
ax[1, 1].title.set_size(18)
ax[1, 1].grid()
ax[1, 1].legend(fontsize=14, loc='upper left')

plt.subplots_adjust(
    left=0.1,    # The left side of the subplots of the figure
    right=0.9,   # The right side of the subplots of the figure
    bottom=0.1,  # The bottom of the subplots of the figure
    top=0.9,     # The top of the subplots of the figure
    wspace=0.2,  # The amount of width reserved for space between subplots
    hspace=0.25   # The amount of height reserved for space between subplots
)

plt.savefig('fig_cifar_tcom.pdf')
plt.show()

# figure - Robustness
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

uniform_salf_path = 'article_vgg11_cifar_uniform_2400_l2'
optimal_salf_path = 'article_vgg11_cifar_salf_2400_l2'
fedavg_path = 'article_vgg11_cifar_fedavg_2400_l2'
drop_path = 'article_vgg11_cifar_drop_2400_l2'

iteration_time_uniform = np.load("checkpoints/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(10, 30, 1, 300)
iteration_time_drop = np.load("checkpoints/" + drop_path + "/iteration_times.npy")

validation_acc_uniform = np.load("checkpoints/" + uniform_salf_path + "/val_acc_list.npy")
validation_acc_optimal = np.load("checkpoints/" + optimal_salf_path + "/val_acc_list.npy")
validation_acc_fedavg = np.load("checkpoints/" + fedavg_path + "/val_acc_list.npy")
validation_acc_drop = np.load("checkpoints/" + drop_path + "/val_acc_list.npy")

time_uniform = np.cumsum(iteration_time_uniform)
time_optimal = np.cumsum(iteration_time_optimal)
time_fedavg = np.cumsum(iteration_time_fedavg)
time_drop = np.cumsum(iteration_time_drop)

ax[0].plot(time_uniform, validation_acc_uniform, label='SALF', marker="o", markevery=20, markersize=6)
ax[0].plot(time_optimal, validation_acc_optimal, label='ADEL-FL', marker="o", markevery=20, markersize=6)
ax[0].plot(time_fedavg, validation_acc_fedavg, label='Wait Stragglers', marker="o", markevery=20, markersize=6)
ax[0].plot(time_drop, validation_acc_drop, label='FedAvg', marker="o", markevery=20, markersize=6)
ax[0].set_xlim([0, 2400])

ax[0].set(
    xlabel='Training time [s]',
    ylabel='Validation Accuracy [%]',
    title='(a) Adding L2 Regularization - CIFAR VGG10'
)

ax[0].xaxis.label.set_size(16)
ax[0].yaxis.label.set_size(16)
ax[0].title.set_size(18)
ax[0].grid()
ax[0].legend(fontsize=12)

# MLP
uniform_salf_path = 'article_vgg11_cifar_uniform_2400_fixlr025'
optimal_salf_path = 'article_vgg11_cifar_salf_2400_fixlr025'
fedavg_path = 'article_vgg11_cifar_fedavg_2400_fixlr025'
drop_path = 'article_vgg11_cifar_drop_2400_fixlr025'

iteration_time_uniform = np.load("checkpoints/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(10, 30, 1, 300)
iteration_time_drop = np.load("checkpoints/" + drop_path + "/iteration_times.npy")

validation_acc_uniform = np.load("checkpoints/" + uniform_salf_path + "/val_acc_list.npy")
validation_acc_optimal = np.load("checkpoints/" + optimal_salf_path + "/val_acc_list.npy")
validation_acc_fedavg = np.load("checkpoints/" + fedavg_path + "/val_acc_list.npy")
validation_acc_drop = np.load("checkpoints/" + drop_path + "/val_acc_list.npy")

time_uniform = np.cumsum(iteration_time_uniform)
time_optimal = np.cumsum(iteration_time_optimal)
time_fedavg = np.cumsum(iteration_time_fedavg)
time_drop = np.cumsum(iteration_time_drop)

ax[1].plot(time_uniform, validation_acc_uniform, label='SALF', marker="o", markevery=20, markersize=6)
ax[1].plot(time_optimal, validation_acc_optimal, label='ADEL-FL', marker="o", markevery=20, markersize=6)
ax[1].plot(time_fedavg, validation_acc_fedavg, label='Wait Stragglers', marker="o", markevery=20, markersize=6)
ax[1].plot(time_drop, validation_acc_drop, label='FedAvg', marker="o", markevery=20, markersize=6)
ax[1].set_xlim([0, 2400])

ax[1].set(
    xlabel='Training time [s]',
    ylabel='Validation Accuracy [%]',
    title='(b) Constant Learning Rate - CIFAR VGG10'
)

ax[1].xaxis.label.set_size(16)
ax[1].yaxis.label.set_size(16)
ax[1].title.set_size(18)
ax[1].grid()
ax[1].legend(fontsize=12)
plt.show()

# figure - NeurIPS
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# MLP
uniform_salf_path = 'article_vgg11_cifar_uniform_momentum'
optimal_salf_path = 'article_vgg11_cifar_fulloptimization_momentum'
fedavg_path = 'article_vgg11_cifar_fedavg_momentum'
drop_path = 'article_vgg11_cifar_drop_momentum'

iteration_time_uniform = np.load("checkpoints/neurips/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/neurips/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(10, 13, 1, 300)
iteration_time_drop = np.load("checkpoints/neurips/" + drop_path + "/iteration_times.npy")

validation_acc_uniform = np.load("checkpoints/neurips/" + uniform_salf_path + "/val_acc_list.npy")
validation_acc_optimal = np.load("checkpoints/neurips/" + optimal_salf_path + "/val_acc_list.npy")
validation_acc_fedavg = np.load("checkpoints/neurips/" + fedavg_path + "/val_acc_list.npy")
validation_acc_drop = np.load("checkpoints/neurips/" + drop_path + "/val_acc_list.npy")

time_uniform = np.cumsum(iteration_time_uniform)
time_optimal = np.cumsum(iteration_time_optimal)
time_fedavg = np.cumsum(iteration_time_fedavg)
time_drop = np.cumsum(iteration_time_drop)

ax[0].plot(time_uniform, validation_acc_uniform, label='SALF', marker="o", markevery=20, markersize=6)
ax[0].plot(time_optimal, validation_acc_optimal, label='ADEL-FL', marker="o", markevery=20, markersize=6)
ax[0].plot(time_fedavg, validation_acc_fedavg, label='Wait Stragglers', marker="o", markevery=20, markersize=6)
ax[0].plot(time_drop, validation_acc_drop, label='FedAvg', marker="o", markevery=20, markersize=6)
ax[0].set_xlim([0, 2400])

ax[0].set(
    xlabel='Training time [s]',
    ylabel='Validation Accuracy [%]',
    title='(a) CIFAR10 VGG11 Convergence'
)

ax[0].xaxis.label.set_size(16)
ax[0].yaxis.label.set_size(16)
ax[0].title.set_size(18)
ax[0].grid()
ax[0].legend(fontsize=12)

# MLP
uniform_salf_path = 'fix_article_vgg13_cifar_uniform_momentum'
optimal_salf_path = 'fix_article_vgg13_cifar_fulloptimization_momentum'
fedavg_path = 'fix_article_vgg13_cifar_fedavg_momentum'
drop_path = 'fix_article_vgg13_cifar_drop_momentum'

iteration_time_uniform = np.load("checkpoints/neurips/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/neurips/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(20, 13, 1, 300)
iteration_time_drop = np.load("checkpoints/neurips/" + drop_path + "/iteration_times.npy")

validation_acc_uniform = np.load("checkpoints/neurips/" + uniform_salf_path + "/val_acc_list.npy")
validation_acc_optimal = np.load("checkpoints/neurips/" + optimal_salf_path + "/val_acc_list.npy")
validation_acc_fedavg = np.load("checkpoints/neurips/" + fedavg_path + "/val_acc_list.npy")
validation_acc_drop = np.load("checkpoints/neurips/" + drop_path + "/val_acc_list.npy")

time_uniform = np.cumsum(iteration_time_uniform)
time_optimal = np.cumsum(iteration_time_optimal)
time_fedavg = np.cumsum(iteration_time_fedavg)
time_drop = np.cumsum(iteration_time_drop)

ax[1].plot(time_uniform, validation_acc_uniform, label='SALF', marker="o", markevery=20, markersize=6)
ax[1].plot(time_optimal, validation_acc_optimal, label='ADEL-FL', marker="o", markevery=20, markersize=6)
ax[1].plot(time_fedavg, validation_acc_fedavg, label='Wait Stragglers', marker="o", markevery=20, markersize=6)
ax[1].plot(time_drop, validation_acc_drop, label='FedAvg', marker="o", markevery=20, markersize=6)
ax[1].set_xlim([0, 3000])

ax[1].set(
    xlabel='Training time [s]',
    ylabel='Validation Accuracy [%]',
    title='(b) CIFAR10 VGG13 Convergence'
)

ax[1].xaxis.label.set_size(16)
ax[1].yaxis.label.set_size(16)
ax[1].title.set_size(18)
ax[1].grid()
ax[1].legend(fontsize=12)
plt.show()

# ---------- FIGURE 1: VGG11 ----------

plt.figure(figsize=(7, 5))

uniform_salf_path = 'article_vgg11_cifar_uniform_momentum'
optimal_salf_path = 'article_vgg11_cifar_fulloptimization_momentum'
fedavg_path = 'article_vgg11_cifar_fedavg_momentum'
drop_path = 'article_vgg11_cifar_drop_momentum'

iteration_time_uniform = np.load("checkpoints/neurips/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/neurips/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(10, 13, 1, 300)
iteration_time_drop = np.load("checkpoints/neurips/" + drop_path + "/iteration_times.npy")

validation_acc_uniform = np.load("checkpoints/neurips/" + uniform_salf_path + "/val_acc_list.npy")
validation_acc_optimal = np.load("checkpoints/neurips/" + optimal_salf_path + "/val_acc_list.npy")
validation_acc_fedavg = np.load("checkpoints/neurips/" + fedavg_path + "/val_acc_list.npy")
validation_acc_drop = np.load("checkpoints/neurips/" + drop_path + "/val_acc_list.npy")

time_uniform = np.cumsum(iteration_time_uniform)
time_optimal = np.cumsum(iteration_time_optimal)
time_fedavg = np.cumsum(iteration_time_fedavg)
time_drop = np.cumsum(iteration_time_drop)

plt.plot(time_uniform, validation_acc_uniform, label='SALF', marker="o", markevery=20, markersize=6)
plt.plot(time_optimal, validation_acc_optimal, label='ADEL-FL', marker="o", markevery=20, markersize=6)
plt.plot(time_fedavg, validation_acc_fedavg, label='Wait Stragglers', marker="o", markevery=20, markersize=6)
plt.plot(time_drop, validation_acc_drop, label='FedAvg', marker="o", markevery=20, markersize=6)

plt.xlim([0, 2400])
plt.xlabel('Training time [s]', fontsize=24)
plt.ylabel('Validation Accuracy [%]', fontsize=24)
#plt.title('(a) CIFAR10 VGG11 Convergence', fontsize=18)
plt.grid()
plt.legend(fontsize=20)
plt.tight_layout()
plt.show()



# ---------- FIGURE 2: VGG13 ----------

plt.figure(figsize=(7, 5))

uniform_salf_path = 'fix_article_vgg13_cifar_uniform_momentum'
optimal_salf_path = 'fix_article_vgg13_cifar_fulloptimization_momentum'
fedavg_path = 'fix_article_vgg13_cifar_fedavg_momentum'
drop_path = 'fix_article_vgg13_cifar_drop_momentum'

iteration_time_uniform = np.load("checkpoints/neurips/" + uniform_salf_path + "/iteration_times.npy")
iteration_time_optimal = np.load("checkpoints/neurips/" + optimal_salf_path + "/iteration_times.npy")
iteration_time_fedavg = fedavg_times.find_fedavg_iteration_time(20, 13, 1, 300)
iteration_time_drop = np.load("checkpoints/neurips/" + drop_path + "/iteration_times.npy")

validation_acc_uniform = np.load("checkpoints/neurips/" + uniform_salf_path + "/val_acc_list.npy")
validation_acc_optimal = np.load("checkpoints/neurips/" + optimal_salf_path + "/val_acc_list.npy")
validation_acc_fedavg = np.load("checkpoints/neurips/" + fedavg_path + "/val_acc_list.npy")
validation_acc_drop = np.load("checkpoints/neurips/" + drop_path + "/val_acc_list.npy")

time_uniform = np.cumsum(iteration_time_uniform)
time_optimal = np.cumsum(iteration_time_optimal)
time_fedavg = np.cumsum(iteration_time_fedavg)
time_drop = np.cumsum(iteration_time_drop)

plt.plot(time_uniform, validation_acc_uniform, label='SALF', marker="o", markevery=20, markersize=6)
plt.plot(time_optimal, validation_acc_optimal, label='ADEL-FL', marker="o", markevery=20, markersize=6)
plt.plot(time_fedavg, validation_acc_fedavg, label='Wait Stragglers', marker="o", markevery=20, markersize=6)
plt.plot(time_drop, validation_acc_drop, label='FedAvg', marker="o", markevery=20, markersize=6)

plt.xlim([0, 3000])
plt.xlabel('Training time [s]', fontsize=24)
plt.ylabel('Validation Accuracy [%]', fontsize=24)
#plt.title('(b) CIFAR10 VGG13 Convergence', fontsize=18)
plt.grid()
plt.legend(fontsize=20)
plt.tight_layout()
plt.show()
