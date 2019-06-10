"""
MI calculation by Histogram method
Reference:  The mutual information: detecting and evaluating 
dependencies between variables by Steuer R
https://doi.org/10.1093/bioinformatics/18.suppl_2.S231
"""
import numpy as np
from matplotlib import pyplot as plt

def make_histogram(X_bin_size, Y_bin_size, X, Y):
    #create bins with given bin-size
    X_bin = [i*0 for i in range(X_bin_size)]
    Y_bin = [i*0 for i in range(Y_bin_size)]
    #normalize Random Variables
    X = (X - np.amin(X)) / (np.amax(X) - np.amin(X))
    Y = (Y - np.amin(Y)) / (np.amax(Y) - np.amin(Y))
    #populate histogram with data points
    for X_i in X:
        X_bin[int(round(X_i,1)*10)%X_bin_size] += 1
    for Y_i in Y:
        Y_bin[int(round(Y_i,1)*10)%Y_bin_size] += 1
    return(X_bin, Y_bin)

def entropy(X):
    #calculate entropy of X, here X is binned by histogram
    p_X = X / np.sum(X)
    return(-1 * np.sum(p_X * np.log(p_X)))

def joint_entropy(X, Y):
    #calculate joint probability of two events
    p_X = X / np.sum(X)
    p_Y = Y / np.sum(Y)
    sum = 0
    for y_i in p_Y:
        sum += np.sum((y_i * p_X) * np.log(y_i * p_X))
    return(-1*sum)

def conditional_entropy(X, Y):
    summation = 0
    for value_x in values_x:
        for value_y in values_y:
            px = shape(where(data[x_index]==value_x))[1] / self.n_cols
            py = shape(where(data[y_index]==value_y))[1] / self.n_cols
            pxy = len(where(in1d(where(data[x_index]==value_x)[0], 
                            where(data[y_index]==value_y)[0])==True)[0]) / self.n_cols
            if pxy > 0.0:
    summation += pxy * math.log((pxy / (px*py)), log_base)

def kullback_divergence(X_bin_size, Y_bin_size, X, Y):
    X_bin, Y_bin = make_histogram(X_bin_size, Y_bin_size, X, Y)
    return(joint_entropy(X_bin, Y_bin) - estimate_mutual_info(X_bin_size, Y_bin_size, X, Y))

def estimate_mutual_info(X_bin_size, Y_bin_size, X, Y):
    X_bin, Y_bin = make_histogram(X_bin_size, Y_bin_size, X, Y)
    return(entropy(X_bin) + entropy(Y_bin) - joint_entropy(X_bin, Y_bin))