import numpy as np
from matplotlib import pyplot as plt
r = np.load('Random_param.npz')
p = np.load('Param_1024_5_divergence.npz')
fig, ax = plt.subplots()
:q1)

ax.boxplot(r['l1'].flatten())

fig, ax = plt.subplots()
ax.boxplot(p['w'].flatten())
plt.show()

