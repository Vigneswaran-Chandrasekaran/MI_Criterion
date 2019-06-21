"""
MI calculation by Histogram method
Reference:  The mutual information: detecting and evaluating 
dependencies between variables by Steuer R
https://doi.org/10.1093/bioinformatics/18.suppl_2.S231
TODO: add other methods and check the robustness
"""
import numpy as np
import time

def estimate_mutual_info(X, neurons, bins = 5):
    xy = np.histogram2d(X, neurons, bins)[0]
    x = np.histogram(X, bins)[0]
    y = np.histogram(neurons, bins)[0]
    ent_x = -1 * np.sum( x / np.sum(x) * np.log( x / np.sum(x)))
    ent_y = -1 * np.sum( y / np.sum(y) * np.log( y / np.sum(y)))
    ent_xy = -1 * np.sum( xy / np.sum(xy) * np.log( xy / np.sum(xy)))
    return (ent_x + ent_y - ent_xy)

tic = time.time()
X = np.random.rand(12000, 1200).astype('float16')
Y = np.random.rand(12000, 10).astype('float16')
for j in Y.T:
    mi = 0
    for i in range(X.shape[1]):
        mi += estimate_mutual_info(X.T[i], j, bins = 2)
    print(mi)
toc = time.time()
print(str(toc - tic)+" seconds")