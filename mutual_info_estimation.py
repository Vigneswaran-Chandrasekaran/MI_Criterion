"""
Mutual Information estimation based on 
non-parametric binning method.I(X|Y) = H(X) - H(X|Y), 
which measure amount of unpredictivity in X decreases 
when Y is known.Direct estimation of I(X|Y) is not 
plausible, hence Kullback-Leibler Divergence based 
estimation is employed.I(X|Y) = D_kl(P_x_y || P_x * P_y), 
where P_x_y is joint distribution and P_x * P_y 
is the product of marginal distributions.The intution is 
quite clearer here, as divergence between joint and 
marginal distributions increases the Information is more, which is 
equivalent to I(X|Y). D_kl(A||B) = E_p[log(dP/dQ)], also 
refer Lebesgue Measure for more accurate calculation. 
Dual representation of KL divergence: 1) Donsker-Varadhan Rrepersentation
                                      2) f-divergence represenation
Here we employ Kernel Estimation (KDE) based 
Mutual Information Estimation, with added noise N(0,sigma**2) to 
make the assumption of Input is mixture of Gaussians
"""
from __future__ import print_function
import keras
import keras.backend as K
import numpy as np

def get_shape(x):
    dims = K.cast( K.shape(x)[1], K.floatx() ) 
    N    = K.cast( K.shape(x)[0], K.floatx() )
    return dims, N

def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = get_shape(x)
    dists = Kget_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims/2.0)*K.log(2*np.pi*var)
    lprobs = K.logsumexp(-dists2, axis=1) - K.log(N) - normconst
    h = -K.mean(lprobs)
    return dims/2 + h

def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    dims, N = get_shape(x)
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2

def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)


noise_variance = 0.05

entropy_func_upper = K.function([Y_samples,], [entropy_estimator_kl(Y_samples, noise_variance),])
entropy_func_lower = K.function([Y_samples,], [entropy_estimator_bd(Y_samples, noise_variance),])

data = np.random.random( size = (1000, 20) )  # N x dims
H_Y_given_X = kde_condentropy(data, noise_variance)
H_Y_upper = entropy_func_upper([data,])[0]
H_Y_lower = entropy_func_lower([data,])[0]

print("Upper bound: %0.3f nats" % (H_Y_upper - H_Y_given_X))
print("Lower bound: %0.3f nats" % (H_Y_lower - H_Y_given_X))

# Alternative calculation, direct from distance matrices
dims, N = get_shape(K.variable(data))
dists = Kget_dists(K.variable(data))
dists2 = dists / (2*noise_variance)
mi2 = K.eval(-K.mean(K.logsumexp(-dists2, axis=1) - K.log(N)))
print("Upper bound2: %0.3f nats" % mi2)


dims, N = get_shape(K.variable(data))
dists = Kget_dists(K.variable(data))
dists2 = dists / (2*4*noise_variance)
mi2 = K.eval(-K.mean(K.logsumexp(-dists2, axis=1) - K.log(N)) )
print("Lower bound2: %0.3f nats" % mi2)

print(Y_samples)