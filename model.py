
import numpy as np
import random
import math
import time
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt

# activation function - instead of tanh or sigmoid
def reLU(z):
	return np.maximum(0, z) 

def loss(desired,final):
    return 0.5*np.sum(desired-final)**2


def conv_forward(X, W, F = 3, S = 1, K=1, P = 1):
    # X = input neurons
    # W = input volume size (shape), F = receptive field size of the Conv Layer neurons
    # S = stride, K = number of filters, P = padding
    height, width, depth = W
    weights = np.random.randn(K, depth, F, F)
    biases = np.random.rand(K, 1)

    out_w = (width - F + 2*P)/S + 1
    out_h = (height - F + 2*P)/S + 1
    out_d = K

    V = np.zeros((K, out_h*out_w))
    out = np.zeros((K, out_h*out_w))

    col_len =  out.shape[1] # = column length (out_h * out_h)

    for i in range(K):
        incr = 0
        row = 0

        for j in range(col_len):  # loop until the output array is filled up -> one dimensional (600)
            V[i][j] = np.sum(X[:,row:F + row, incr:F + incr] * weights[i]) + biases[i]
            out[i][j] = reLU(V[i][j])
            incr += stride

            if (F + incr)-S >= width:  # wrap indices at the end of each row
                incr = 0
                row += S

    V = V.reshape((K, out_h, out_w))
    out = out.reshape((K, out_h, out_w))
    return V, weights, out


def max_pool_forward(X, W, F = 2, S = 2):
    # W = input volume size, F = spatial extent, S = stride
    inheight, inwidth, indepth = W
    pool_height, pool_width = F, S

    out_h = (inheight - F) / S + 1
    out_w = (inwidth - F) / S + 1

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

def affine_forward(X, W, out_sz):
    # if last layer, then out_sz is the number of classes
    inheight, inwidth, depth = W
    weights = np.random.randn(out_sz, depth*inheight*inwidth)
    biases = np.random.randn(out_sz)
    weights.reshape((out_sz, depth, inheight, inwidth))
    X = X.reshape((depth * inheight * inwidth, 1))
    V = np.dot(X, weights) + biases
    out = reLU(V)
    return V, out

class Model:

    def __init__(self, W, layers):

    	self.W = W
        self.layers = layers
        self.layer_cache = []
    	self.layer_weight_shapes = []
        self.layer_biases_shapes = []

    def feedforward(self, image):
        prev_out = image

        # forward pass
        for layer in self.layers:
            X = prev_out
            W = prev_out.shape

            if layer == "conv":
                V, weights, out = conv_forward(X, W, 3, 1, 1, 1)
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
                V, out = affine_forward(X, W, h*w*d)
            elif layer == "final":
                # 2 classes
                V, out = affine_forward(X, W, 2)
            else:
                raise NotImplementedError

            layer_cache.append(V)
            prev_out = out

        final_activation = prev_activation
        return final_activation

    def backpropagate(self, image, label):
        



