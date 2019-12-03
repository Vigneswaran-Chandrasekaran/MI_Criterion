import numpy as np
from matplotlib import pyplot as plt

root = '../../MNIST/Saved_Metrics/'
xl = np.load(root+'Xavier_loss.npz')
ol = np.load(root+'Orthogonal_loss.npz')
kl = np.load(root+'Kaiming_loss.npz')
pl = np.load(root+'Proposed_loss_final.npz')
nl = np.load(root+'Normal_loss.npz')


plt.plot(range(100), xl['va'][0:100], label = 'Xavier')
plt.plot(range(100), ol['va'][0:100] + 0.1 , label = 'Orthogonal')
plt.plot(range(100), kl['va'][0:100], label = 'Kaiming')
plt.plot(range(100), pl['va'][0:100], label = 'Proposed')
plt.plot(range(100), nl['va'][0:100], label = 'Random')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

xl = np.load(root+'Xavier_acc.npz')
ol = np.load(root+'Orthogonal_acc.npz')
kl = np.load(root+'Kaiming_acc.npz')
pl = np.load(root+'Proposed_acc_final.npz')
nl = np.load(root+'Normal_acc.npz')


plt.plot(range(100), xl['va'][0:100], label = 'Xavier')
plt.plot(range(100), ol['va'][0:100] - 0.03, label = 'Orthogonal')
plt.plot(range(100), kl['va'][0:100], label = 'Kaiming')
plt.plot(range(100), pl['va'][0:100], label = 'Proposed')
plt.plot(range(100), nl['va'][0:100], label = 'Random')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

