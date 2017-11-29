
import numpy as np
import random
import math
import time
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt

# activation function - instead of tanh or sigmoid

def loss(desired,final):
    return 0.5*np.sum(desired-final)**2


def conv_forward(X, F = 3, S = 1, K=1, P = 1):
    # X = input neurons
    # F = receptive field size of the Conv Layer neurons
    # S = stride, K = number of filters, P = padding
    n_x, d_x, h_x, w_x = X.shape
    weights = np.random.randn(K, depth, F, F)
    biases = np.random.rand(K, 1)

    out_w = (width - F + 2*P)/S + 1
    out_h = (height - F + 2*P)/S + 1
    out_d = K

    x_col = im2col_indices(X, F, F, padding=P, stride=S)
    W_row = weights.reshape(K, -1)

    out = np.dot(w_row, x_col) + biases 
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, weights, biases, S, P, x_col)
    return cache, out
"""
    col_len =  out.shape[1] # = column length (out_h * out_h)
    xcols = []

    for i in range(K):
        incr = 0
        row = 0

        for j in range(col_len):  # loop until the output array is filled up -> one dimensional (600)
            im_2_col = X[:,row:F + row, incr:F + incr]
            xcols.append(im_2_col)
            V[i][j] = np.sum(im_2_col * weights[i]) + biases[i]
            out[i][j] = reLU(V[i][j])
            incr += stride

            if (F + incr)-S >= width:  # wrap indices at the end of each row
                incr = 0
                row += S

    V = V.reshape((K, out_h, out_w))
    out = out.reshape((K, out_h, out_w))
    return V, weights, biases, out, xcols
"""


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
"""
    X_r = X.reshape(n_x * d_x, 1, h_x, w_x)
    x_col = im2col_indices(X_r, F, F, padding=0, stride=S)

    indxs = np.argmax(x_col, axis=0)
    maxs = x_col[indxs, range(indxs.size)]

    maxs = maxs.reshape(h_out, w_out, n, d)
    maxs = maxs.transpose(2, 3, 0, 1)

    cache = (X, F, S, x_col, indxs)
    return maxs, cache

 #####
    out = np.empty((indepth, out_h * out_w))
    indxs = np.empty((indepth, out_h * out_w, 2))

    # for each filter pool
    for i in range(indepth):
        row = 0
        incr = 0
        for j in range(out_h * out_w):
            block = X[i][row:self.F + row, slide:self.S + incr]
            out[i][j] = np.amax(block)
            indx = zip(*np.where(block == np.amax(block)))
            if len(indx) > 1:
                indx = [indx[0]]
            indxs[i][j] = indx[0][0]+row, indx[0][1]+incr
            incr += S 

            if incr >= inwidth:
                incr = 0
                row += poolheight
    out = out.reshape(indepth, out_h, out_w)
    indxs = indxs.reshape(indepth, out_h, out_w, 2)
    return indxs, out
    """

def affine_forward(x, w, b):
    # if last layer, then out_sz is the number of classes
    out = np.dot(X, W) + b
    cache = (W, X)
    return cache, out
"""
    n_x, d_x, h_x, w_x = X.shape
    weights = np.random.randn(out_sz, h_d*h_x*h_w)
    biases = np.random.randn(out_sz)
    weights.reshape((out_sz, depth, inheight, inwidth))
    X = X.reshape((depth * inheight * inwidth, 1))
    V = np.dot(X, weights) + biases
    out = reLU(V)
    return V, weights, biases, out
"""
# activation function - instead of tanh or sigmoid
def relu_forward(X):
    out = np.maximum(X, 0)
    cache = X
    return out, cache


class Model:

    def __init__(self, W, layers):

    	self.W = W
        self.layers = layers
        self.layer_cache = []
        self.layer_out_cache = []
    	self.layer_weight_shapes = []
        self.layer_biases_shapes = []

    def feedforward(self, image):
        prev_out = image

        # forward pass
        for layer in self.layers:
            X = prev_out
            W = prev_out.shape

            if layer == "conv":
                V, weights, bias, out = conv_forward(X, W, 3, 1, 1, 1)
                for i in range(out.shape[0]):
                    plt.imsave('images/cat_conv%d.jpg'%i, out[i])
                for i in range(weights.shape[0]):
                    plt.imsave('images/filter_conv%s.jpg'%i, weights[i].reshape((5,5)))
            elif layer == "pool":
                V, out = max_pool_forward(X, W, 2, 2)
                for i in range(out.shape[0]):
                    plt.imsave('images/pool_pic%s.jpg'%i, out[i])
            elif layer == "fc":
                h, w, d = W
                V, weights, bias, out = affine_forward(X, W, h*w*d)
            elif layer == "final":
                # 2 classes
                V, weights, bias, out = affine_forward(X, W, 2)
            else:
                raise NotImplementedError

            self.layer_cache.append(V)
            self.layer_out_cache.append(out)
            if layer != 'pool':
                self.layer_weight_shapes.append(weights.shape)
                self.layer_biases_shapes.append(bias.shape)
            prev_out = out

        final_activation = prev_activation
        return final_activation

    def backpropagate(self, image, label):
        





