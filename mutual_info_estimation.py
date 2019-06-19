"""
MI calculation by Histogram method
Reference:  The mutual information: detecting and evaluating 
dependencies between variables by Steuer R
https://doi.org/10.1093/bioinformatics/18.suppl_2.S231
TODO: add other methods and check the robustness
"""

import numpy as np
import time

tic = time.time()
def estimate_mutual_info(X, neurons, bins = 5):
    neuronal_MI = np.zeros(neurons.shape[1])
    index = 0
    for Y in neurons.T:
        sum = 0
        for dim in range(X.shape[1]):
            xy = np.histogram2d(X[:,dim], Y, bins)[0]
            x = np.histogram(X[:,dim], bins)[0]
            y = np.histogram(Y, bins)[0]
            ent_x = -1 * np.sum( x / np.sum(x) * np.log( x / np.sum(x)))
            ent_y = -1 * np.sum( y / np.sum(y) * np.log( y / np.sum(y)))
            ent_xy = -1 * np.sum( xy / np.sum(xy) * np.log( xy / np.sum(xy)))
            sum +=  ent_x + ent_y - ent_xy
        neuronal_MI[index] += sum
        index += 1
    return(neuronal_MI)

X = np.random.rand(12000, 1200)
Y = np.random.rand(12000, 10)

print(estimate_mutual_info(X,Y))