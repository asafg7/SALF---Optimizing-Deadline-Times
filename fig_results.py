import numpy as np
import matplotlib.pyplot as plt
import fedavg_times

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Sub Figure a - deadline allocation

path1 = 'tmp_article_vgg11_cifar_uniform_none'
path2 = 'tmp_article_vgg11_cifar_fulloptimization_none'

label1 = 'SALF'
label2 = 'ADEL-FL'

iteration_time_1 = np.load("checkpoints/" + path1 + "/iteration_times.npy")
iteration_time_2 = np.load("checkpoints/" + path2 + "/iteration_times.npy")

ax[0].plot(range(len(iteration_time_1)), iteration_time_1, label=label1)
ax[0].plot(range(len(iteration_time_2)), iteration_time_2, label=label2)

ax[0].set(xlabel='Round \n (a) deadline allocation for VGG11 CIFAR10',
          ylabel='deadline time')
ax[0].grid()
ax[0].legend()

# Sub Figure b - MLP MNIST

path1 = 'article_mlp_mnist_uniform_default'
path2 = 'article_mlp_mnist_fulloptimization_default'
path3 = 'article_mlp_mnist_fedavg_default'
path4 = 'article_mlp_mnist_drop_default'

label1 = 'SALF'
label2 = 'ADEL-FL'
label3 = 'Wait Stragglers'
label4 = 'FedAvg'

iteration_time_1 = np.load("checkpoints/" + path1 + "/iteration_times.npy")
iteration_time_2 = np.load("checkpoints/" + path2 + "/iteration_times.npy")
iteration_time_3 = fedavg_times.find_fedavg_iteration_time(20, 11, 1, 150)
iteration_time_4 = np.load("checkpoints/" + path4 + "/iteration_times.npy")

validation_1 = np.load("checkpoints/" + path1 + "/val_acc_list.npy")
validation_2 = np.load("checkpoints/" + path2 + "/val_acc_list.npy")
validation_3 = np.load("checkpoints/" + path3 + "/val_acc_list.npy")
validation_4 = np.load("checkpoints/" + path4 + "/val_acc_list.npy")

time_1 = np.cumsum(iteration_time_1)
time_2 = np.cumsum(iteration_time_2)
time_3 = np.cumsum(iteration_time_3)
time_4 = np.cumsum(iteration_time_4)

ax[1].plot(time_1, validation_1, label=label1, marker="o", markevery=20, markersize=6)
ax[1].plot(time_2, validation_2, label=label2, marker="o", markevery=20, markersize=6)
ax[1].plot(time_3, validation_3, label=label3, marker="o", markevery=20, markersize=6)
ax[1].plot(time_4, validation_4, label=label4, marker="o", markevery=20, markersize=6)

ax[1].set(xlabel='Training Time \n (b) MNIST MLP convergence',
          ylabel='Validation Accuracy',
          xlim=[0, 400])
ax[1].grid()
ax[1].legend()

# Sub Figure c - CNN MNIST

path1 = 'article_cnn_mnist_uniform_default'
path2 = 'article_cnn_mnist_fulloptimization_default'
path3 = 'article_cnn_mnist_fedavg_default'
path4 = 'article_cnn_mnist_drop_default'

label1 = 'SALF'
label2 = 'ADEL-FL'
label3 = 'Wait Stragglers'
label4 = 'FedAvg'

iteration_time_1 = np.load("checkpoints/" + path1 + "/iteration_times.npy")
iteration_time_2 = np.load("checkpoints/" + path2 + "/iteration_times.npy")
iteration_time_3 = fedavg_times.find_fedavg_iteration_time(20, 11, 1, 50)
iteration_time_4 = np.load("checkpoints/" + path4 + "/iteration_times.npy")

validation_1 = np.load("checkpoints/" + path1 + "/val_acc_list.npy")
validation_2 = np.load("checkpoints/" + path2 + "/val_acc_list.npy")
validation_3 = np.load("checkpoints/" + path3 + "/val_acc_list.npy")
validation_4 = np.load("checkpoints/" + path4 + "/val_acc_list.npy")

time_1 = np.cumsum(iteration_time_1)
time_2 = np.cumsum(iteration_time_2)
time_3 = np.cumsum(iteration_time_3)
time_4 = np.cumsum(iteration_time_4)

ax[2].plot(time_1, validation_1, label=label1, marker="o", markevery=20, markersize=6)
ax[2].plot(time_2, validation_2, label=label2, marker="o", markevery=20, markersize=6)
ax[2].plot(time_3, validation_3, label=label3, marker="o", markevery=20, markersize=6)
ax[2].plot(time_4, validation_4, label=label4, marker="o", markevery=20, markersize=6)

ax[2].set(xlabel='Training Time \n (b) MNIST CNN convergence',
          ylabel='Validation Accuracy',
          xlim=[0, 150])
ax[2].grid()
ax[2].legend()

# end
plt.tight_layout()
plt.show()