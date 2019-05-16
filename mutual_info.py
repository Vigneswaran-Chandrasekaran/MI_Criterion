import numpy as np
from scipy.special import gamma,psi
from scipy import ndimage
from scipy.linalg import det
from numpy import pi
EPS = np.finfo(float).eps

def mutual_information_2d(x, y, sigma=1, normalized=False):
    bins = (256, 256)
    jh = np.histogram2d(x, y, bins=bins)[0]
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))- np.sum(s2 * np.log(s2)))
    return mi

a = np.random.rand(100, 1000)
label = np.zeros((100, 1000))
mut = 0
for i in range(a.shape[0]):
    mut += mutual_information_2d(a[i], label[i])
print(mut)
