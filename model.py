
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

class discriminator:

    def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=5,
                 hidden_dim=500, num_classes=10, weight_scale=1e-3, reg=1e-3,
                 dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        self.K = num_filters
        self.F = filter_size
        self.S = 1
        self.P = (filter_size - 1)/2

        D, H, W = input_dim
        # conv layer 1: This finds 32 different 5 x 5 pixel features
        self.params['W1'] = np.random.normal(0, weight_scale, (self.K, D, self.F, self.F))
        self.params['b1'] = np.zeros(self.K)
        # pool layer 1
        conv1_h = (H - self.F + 2*self.P)/self.S + 1
        conv1_w = (W - self.F + 2*self.P)/self.S + 1

        pool1_sz = 2
        pool1_h = conv1_h/pool1_sz
        pool1_w = conv1_w/pool1_sz

        # conv layer 2: This finds 64 different 5 x 5 pixel features
        self.params['W2'] = np.random.normal(0, weight_scale, (64, 32, self.F, self.F))
        self.params['b2'] = np.zeros(64)
        # pool layer 2
        conv2_h = (pool1_h - self.F + 2*self.P)/self.S + 1
        conv2_w = (pool1_h - self.F + 2*self.P)/self.S + 1

        pool2_sz = 2
        pool2_h = conv2_h/pool2_sz
        pool2_w = conv2_w/pool2_sz

        # hidden fc layer
        self.params['W3'] = np.random.normal(0, weight_scale, (64*pool2_h*pool2_w, hidden_dim))
        self.params['b3'] = np.zeros(hidden_dim)
        # classifying fc layer
        self.params['W4'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b4'] = np.zeros(num_classes)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)



    def loss(self, X, y=None):
        # Evaluate loss and gradient for the three-layer convolutional network.
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # feed forward
        # conv->relu layer 1
        conv1_cache, conv1_out = conv_forward(X, W1, b1, self.F, self.S, self.K, self.P)
        relu1_cache, relu1_out = relu_forward(conv1_out)
        # pool layer 1
        pool1_cache, pool1_out = max_pool_forward(relu1_out, 2, 2)
        # conv->relu layer 2
        conv2_cache, conv2_out = conv_forward(pool1_out, W2, b2, self.F, self.S, 64, self.P)
        relu2_cache, relu2_out = relu_forward(conv2_out)
        # pool layer 2
        pool2_cache, pool2_out = max_pool_forward(relu2_out, 2, 2)
        # FC->relu layer 1
        fc1_cache, fc1_out = fc_forward(pool2_out, W3, b3)
        relu3_cache, relu3_out = relu_forward(fc1_out)
        #FC 2 - classifying layer
        fc2_cache, fc2_out = fc_forward(relu3_out, W4, b4)
        scores = fc2_out

        if y is None:
          return scores
        
        loss, grads = 0, {}

        # backpropagation

        loss, dscores = softmax_loss(scores, y)
        loss += L2_regularize(self.reg, W1)
        loss += L2_regularize(self.reg, W2)
        loss += L2_regularize(self.reg, W3)
        loss += L2_regularize(self.reg, W4)

        fc2_dx, fc2_dw, fc2_db = bp_fc(dscores, fc2_cache)
        grads['W4'] = fc2_dw + self.reg * self.params['W4']
        grads['b4'] = fc2_db

        relu3_dx = bp_relu(fc2_dx, relu3_cache)

        fc1_dx, fc1_dw, fc1_db = bp_fc(relu3_dx, fc1_cache)
        grads['W3'] = fc1_dw + self.reg * self.params['W3']
        grads['b3'] = fc1_db

        pool2_dx = bp_pool(fc1_dx, pool2_cache)
        relu2_dx = bp_relu(pool2_dx, relu2_cache)

        conv2_dx, conv2_dw, conv2_db = bp_conv(relu2_dx, conv2_cache)
        grads['W2'] = conv2_dw + self.reg * self.params['W2']
        grads['b2'] = conv2_db

        pool1_dx = bp_pool(conv2_dx, pool1_cache)
        relu1_dx = bp_relu(pool1_dx, relu1_cache)

        conv1_dx, conv1_dw, conv1_db = bp_conv(relu1_dx, conv1_cache)
        grads['W1'] = conv1_dw + self.reg * self.params['W1']
        grads['b1'] = conv1_db

        return loss, grads






