
import numpy as np
import random
import math
import time
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from im2col import *

def loss(desired,final):
    return 0.5*np.sum(desired-final)**2


def conv_forward(X, weights, biases, F = 5, S = 1, K=1, P = 1):
    # X = input neurons
    # F = receptive field size of the Conv Layer neurons
    # S = stride, K = number of filters, P = padding
    n_x, d_x, h_x, w_x = X.shape

    out_w = (w_x - F + 2*P)/S + 1
    out_h = (h_x - F + 2*P)/S + 1
    out_d = K

    x_col = im2col_indices(X, F, F, padding=P, stride=S)
    w_row = weights.reshape(K, -1)

    out = np.dot(w_row, x_col) + biases.reshape(-1, 1)
    out = out.reshape(K, out_h, out_w, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, weights, biases, S, P, x_col)
    return cache, out


def max_pool_forward(X, F = 2, S = 2):
    # F = spatial extent, S = stride
    n_x, d_x, h_x, w_x = X.shape
    pool_height, pool_width = F, S

    out_h = (h_x - F) / S + 1
    out_w = (w_x - F) / S + 1

    maxs = np.zeros((n_x, d_x, out_h, out_w))

    for n in range(n_x):
        for d in range(d_x):
            for h in range(out_h):
                for w in range(out_w):
                    pool = X[n, d, h*S:h*S+F, w*S:w*S+F]
                    pool_max = np.amax(pool)
                    maxs[n,d,h,w] = pool_max

    cache = (X,F,S)
    return cache, maxs


def fc_forward(X, W, b):
    N = X.shape[0]
    x_r = X.reshape(N,-1)
    out = np.dot(x_r, W) + b
    cache = (W, X)
    return cache, out

# activation function - instead of tanh or sigmoid
def relu_forward(X):
    out = np.maximum(X, 0)
    cache = X
    return cache, out


def bn_forward(X, gamma, beta, running_mean, running_var, momentum, is_testing = False):
    if is_testing:
        mu = running_mean
        var = running_var
    else:
        mu = np.mean(X, axis = 0)
        var = np.var(X, axis = 0)

    eps = 1e-4*np.ones(var.shape)

    X_norm = (X - mu) / np.sqrt(var + eps)
    # print X_norm.shape, gamma.shape, beta.shape
    X_out = gamma * X_norm + beta
    
    if not is_testing:
        running_mean *= momentum
        running_mean += (1 - momentum) * mu

        running_var *= momentum
        running_var += (1 - momentum) * var

    cache = (gamma, beta, mu, var, eps, X_norm)

    return cache, X_out, running_mean, running_var


