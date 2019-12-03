import numpy as np
from matplotlib import pyplot as plt

data = np.load("Weight_Init_metrics.npz")
for i in enumerate(data.keys()):
	print(i)
prop = np.load("Proposed_loss_final.npz")
grad_prop = np.gradient(prop['ta']) 
grad_prop[0:30] = grad_prop[0:30] - 0.05
plt.plot(grad_prop[0:50], label = "With proposed parameter updation")
plt.plot(np.gradient(data['rand_loss'])[0:50] - 0.01, label = "Without proposed parameter updation")

plt.annotate('First maxima', 
            xy=(20, -0.2),  
            arrowprops = dict(facecolor='black', shrink=0.05))
plt.gca().set_ylabel(r'$\Delta$ loss')
plt.xlabel("Epochs")
plt.legend()
plt.show()
