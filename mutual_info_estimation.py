from matplotlib import pyplot as plt
import numpy as np

prop = np.load("Proposed_accuracy_200_epochs.npz")
normal = np.load("Normal_accuracy_monitor.npz")

plt.plot(range(0,100), prop['va'][0:100], label = ('Binwise pretraining val acc'))
plt.plot(range(0,100), normal['va'][0:100], label = ('Random Init val acc'))
plt.plot(range(0,100), prop['ta'][0:100], label = ('Binwise pretraining train acc'))
plt.plot(range(0,100), normal['ta'][0:100], label = ('Random Init train acc'))

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()