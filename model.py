
import numpy as np
import random
import math
import time
from feedforward import * 
from backprop import *
from im2col import *
import sys

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

    def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=5,
                 hidden_dim=500, num_classes=10, weight_scale=1e-3, reg=1e-3,
                 momentum = 0.9, dtype=np.float32):
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

        # Batch Normalization Parameters
        self.params['gamma1'] = np.random.normal(1, 0.1, (32, H, W))
        self.params['beta1'] = np.random.normal(0, weight_scale, (32, H, W))
        self.running_mean1 = np.zeros((32, H, W))
        self.running_var1 = np.zeros((32, H, W))
        self.params['gamma2'] = np.random.normal(1, 0.1, (64, conv2_h, conv2_w))
        self.params['beta2'] = np.random.normal(0, weight_scale, (64, conv2_h, conv2_w))
        self.running_mean2 = np.zeros((64, conv2_h, conv2_w))
        self.running_var2 = np.zeros((64, conv2_h, conv2_w))
        self.params['gamma3'] = np.random.normal(1, 0.1, (hidden_dim))
        self.params['beta3'] = np.random.normal(0, weight_scale, hidden_dim)
        self.running_mean3 = np.zeros(hidden_dim)
        self.running_var3 = np.zeros(hidden_dim)
        self.momentum = momentum

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None, is_testing = False):
        # Evaluate loss and gradient for the three-layer convolutional network.
        W1, b1 = self.params['W1'], self.params['b1']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        running_mean1, running_var1 = self.running_mean1, self.running_var1
        
        W2, b2 = self.params['W2'], self.params['b2']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        running_mean2, running_var2 = self.running_mean2, self.running_var2
        
        W3, b3 = self.params['W3'], self.params['b3']
        gamma3, beta3 = self.params['gamma3'], self.params['beta3']
        running_mean3, running_var3 = self.running_mean3, self.running_var3
        
        W4, b4 = self.params['W4'], self.params['b4']

        # feed forward
        # conv->relu layer 1
        conv1_cache, conv1_out = conv_forward(X, W1, b1, self.F, self.S, self.K, self.P)
        
        # Batch normalize
        bn1_cache, bn1_out, run_mean1, run_var1 = bn_forward(conv1_out, gamma1, beta1, 
                                     running_mean1, running_var1, self.momentum, is_testing)
        self.running_mean1, self.running_var1 = run_mean1, run_var1
        # Activation
        relu1_cache, relu1_out = relu_forward(bn1_out)
        # pool layer 1
        pool1_cache, pool1_out = max_pool_forward(relu1_out, 2, 2)

        # conv->relu layer 2
        conv2_cache, conv2_out = conv_forward(pool1_out, W2, b2, self.F, self.S, 64, self.P)
        # Batch normalize
        bn2_cache, bn2_out, run_mean2, run_var2 = bn_forward(conv2_out, gamma2, beta2, 
                                     running_mean2, running_var2, self.momentum, is_testing)
        self.running_mean2, self.running_var2 = run_mean2, run_var2
        # Activation
        relu2_cache, relu2_out = relu_forward(bn2_out)
        # pool layer 2
        pool2_cache, pool2_out = max_pool_forward(relu2_out, 2, 2)
        
        # FC->relu layer 1
        fc1_cache, fc1_out = fc_forward(pool2_out, W3, b3)
        # Batch normalize
        bn3_cache, bn3_out, run_mean3, run_var3 = bn_forward(fc1_out, gamma3, beta3, 
                                     running_mean3, running_var3, self.momentum, is_testing)
        self.running_mean3, self.running_var3 = run_mean3, run_var3
        # Activation
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
                                                      ###
        bn3_dx, bn3_dgamma, bn3_dbeta = bp_batchnorm(relu3_dx, bn3_cache)
        grads['gamma3'] = bn3_dgamma #+ self.reg * self.params['gamma3']
        grads['beta3'] = bn3_dbeta
                                        ###
        fc1_dx, fc1_dw, fc1_db = bp_fc(bn3_dx, fc1_cache)
        grads['W3'] = fc1_dw + self.reg * self.params['W3']
        grads['b3'] = fc1_db

        pool2_dx = bp_pool(fc1_dx, pool2_cache)
        relu2_dx = bp_relu(pool2_dx, relu2_cache)
                                                      ###
        bn2_dx, bn2_dgamma, bn2_dbeta = bp_batchnorm(relu2_dx, bn2_cache)
        grads['gamma2'] = bn2_dgamma #+ self.reg * self.params['gamma2']
        grads['beta2'] = bn2_dbeta
                                                ####
        conv2_dx, conv2_dw, conv2_db = bp_conv(bn2_dx, conv2_cache)
        grads['W2'] = conv2_dw + self.reg * self.params['W2']
        grads['b2'] = conv2_db

        pool1_dx = bp_pool(conv2_dx, pool1_cache)
        relu1_dx = bp_relu(pool1_dx, relu1_cache)
                                                      ###
        bn1_dx, bn1_dgamma, bn1_dbeta = bp_batchnorm(relu1_dx, bn1_cache)
        grads['gamma1'] = bn1_dgamma #+ self.reg * self.params['gamma1']
        grads['beta1'] = bn1_dbeta
                                                ###
        conv1_dx, conv1_dw, conv1_db = bp_conv(bn1_dx, conv1_cache)
        grads['W1'] = conv1_dw + self.reg * self.params['W1']
        grads['b1'] = conv1_db
        # raise(ValueError())
        sys.stdout.write(str(loss) + "\n")
        sys.stdout.flush()
        return loss, grads



