import numpy as np
import matplotlib.pyplot as plt

trivial_allocation = np.load("val_acc_list_triv.npy")
optimal_allocation = np.load("val_acc_list_opt.npy")

iter = np.arange(1,np.size(optimal_allocation)+1)

fig, ax = plt.subplots()
ax.plot(iter, np.c_[trivial_allocation, optimal_allocation], label=['Trivial', 'Optimal'])

ax.set(xlabel='Iteration', ylabel='Validation Accuracy',
       title='Trivial vs Optimal Iteration Time Allocation')
ax.grid()
plt.legend();
plt.show()
