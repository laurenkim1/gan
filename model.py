
import numpy as np
import random
import math
import time
from feedforward import * 
from backprop import *
from im2col import *

def L2_regularize(reg, weights):
    return 0.5*reg*np.sum(weights**2)

def softmax_loss(x, y):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

class CNN:

    def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=3,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        self.K = num_filters
        self.F = filter_size
        self.S = 1
        self.P = (filter_size - 1)/2

        D, H, W = input_dim
        # conv layer
        self.params['W1'] = np.random.normal(0, weight_scale, (self.K, D, self.F, self.F))
        self.params['b1'] = np.zeros(self.K)
        # pool layer
        conv_h = (H - self.F + 2*self.P)/self.S + 1
        conv_w = (W - self.F + 2*self.P)/self.S + 1

        pool_h = conv_h/2#(conv_h - self.F) / 2 + 1
        pool_w = conv_w/2#(conv_w - self.F) / 2 + 1
        # hidden affine layer
        self.params['W2'] = np.random.normal(0, weight_scale, (self.K*pool_h*pool_w, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        # classifying affine layer
        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)



    def loss(self, X, y=None):
        # Evaluate loss and gradient for the three-layer convolutional network.
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # feed forward

        conv_cache, conv_out = conv_forward(X, W1, b1, self.F, self.S, self.K, self.P)
        relu1_cache, relu1_out = relu_forward(conv_out)

        pool_cache, pool_out = max_pool_forward(relu1_out, 2, 2)

        affine1_cache, affine1_out = affine_forward(pool_out, W2, b2)
        relu2_cache, relu2_out = relu_forward(affine1_out)

        affine2_cache, affine2_out = affine_forward(relu2_out, W3, b3)
        scores = affine2_out

        if y is None:
          return scores
        
        loss, grads = 0, {}

        # backpropagation

        loss, dscores = softmax_loss(scores, y)
        loss += L2_regularize(self.reg, W1)
        loss += L2_regularize(self.reg, W2)
        loss += L2_regularize(self.reg, W3)

        affine2_dx, affine2_dw, affine2_db = bp_fc(dscores, affine2_cache)
        grads['W3'] = affine2_dw + self.reg * self.params['W3']
        grads['b3'] = affine2_db

        relu2_dx = bp_relu(affine2_dx, relu2_cache)

        affine1_dx, affine1_dw, affine1_db = bp_fc(relu2_dx, affine1_cache)
        grads['W2'] = affine1_dw + self.reg * self.params['W2']
        grads['b2'] = affine1_db

        pool_dx = bp_pool(affine1_dx, pool_cache)
        relu1_dx = bp_relu(pool_dx, relu1_cache)

        conv_dx, conv_dw, conv_db = bp_conv(relu1_dx, conv_cache)
        grads['W1'] = conv_dw + self.reg * self.params['W1']
        grads['b1'] = conv_db

        return loss, grads






