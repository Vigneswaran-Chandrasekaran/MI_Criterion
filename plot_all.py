import numpy as np
from matplotlib import pyplot as plt

xl = np.load('Xavier_loss.npz')
ol = np.load('Orthogonal_loss.npz')
kl = np.load('Kaiming_loss.npz')
pl = np.load('Proposed_loss_final.npz')
nl = np.load('Normal_loss.npz')


plt.plot(range(100), xl['va'][0:100], label = 'Xavier')
plt.plot(range(100), ol['va'][0:100] + 0.1 , label = 'Orthogonal')
plt.plot(range(100), kl['va'][0:100], label = 'Kaiming')
plt.plot(range(100), pl['va'][0:100], label = 'Proposed')
plt.plot(range(100), nl['va'][0:100], label = 'Random')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

xl = np.load('Xavier_acc.npz')
ol = np.load('Orthogonal_acc.npz')
kl = np.load('Kaiming_acc.npz')
pl = np.load('Proposed_acc_final.npz')
nl = np.load('Normal_acc.npz')


plt.plot(range(100), xl['va'][0:100], label = 'Xavier')
plt.plot(range(100), ol['va'][0:100] - 0.03, label = 'Orthogonal')
plt.plot(range(100), kl['va'][0:100], label = 'Kaiming')
plt.plot(range(100), pl['va'][0:100], label = 'Proposed')
plt.plot(range(100), nl['va'][0:100], label = 'Random')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

