import numpy as np
import random
import math
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


class generator:
	def __init__(self, z_dim=(100), num_filters=32, filter_size=3,
			hidden_dim=500, num_classes=10, weight_scale=1e-3, reg=1e-3,
			dtype=np.float32):
		self.params = {}
		self.reg = reg
		self.dtype = dtype
		self.z_dim = z_dim

		self.K = num_filters
		self.F = filter_size
		self.S = 1
		self.P = 1

		# FC 1
		self.params['W1'] = np.random.normal(0, weight_scale, (self.z_dim, 3249))
		self.params['b1'] = np.zeros(3249)

		# conv layer 1: generate 50 features
		self.params['W2'] = np.random.normal(0, weight_scale, (self.z_dim/2, 1, 3, 3))
		self.params['b2'] = np.zeros(self.z_dim/2)

		# conv layer 2: generate 25 features
		self.params['W3'] = np.random.normal(0, weight_scale, (self.z_dim/4, self.z_dim/2, 3, 3))
		self.params['b3'] = np.zeros(self.z_dim/4)

		# conv layer 3: final convolution with one output channel
		self.params['W4'] = np.random.normal(0, weight_scale, (1, self.z_dim/4, 3, 3))
		self.params['b4'] = np.zeros(1)

		for k, v in self.params.iteritems():
			self.params[k] = v.astype(dtype)


	def get_new(self, X, y=None):
		# Evaluate loss and gradient for the three-layer convolutional network.
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		W3, b3 = self.params['W3'], self.params['b3']
		W4, b4 = self.params['W4'], self.params['b4']

		# feed forward

		# fc layer 1
		fc_cache, fc_out = fc_forward(X, W1, b1)
		relu1_cache, relu1_out = relu_forward(fc_out)
		r_relu1_out = relu1_out.reshape((-1, 1, 57, 57))

		# conv->relu layer 1
		conv1_cache, conv1_out = conv_forward(r_relu1_out, W2, b2, self.F, self.S, self.z_dim/2, self.P)
		relu2_cache, relu2_out = relu_forward(conv1_out)
		# conv->relu layer 2
		conv2_cache, conv2_out = conv_forward(relu2_out, W3, b3, self.F, self.S, self.z_dim/4, self.P)
		relu3_cache, relu3_out = relu_forward(conv2_out)
		# conv->sigmoid layer 3
		conv3_cache, conv3_out = conv_forward(relu3_out, W4, b4, self.F, 2, 1, 0)
		sigmoid_cache, sigmoid_out = relu_forward(conv3_out)

		image = sigmoid_out

		if y is None:
			return image.shape
		"""
		loss, grads = 0, {}

		print "a"

		# backpropagation

		loss, dimage = softmax_loss(image, y)
		print loss
		print dimage
		loss += L2_regularize(self.reg, W1)
		loss += L2_regularize(self.reg, W2)
		loss += L2_regularize(self.reg, W3)
		loss += L2_regularize(self.reg, W4)

		sigmoid_dx = bp_sigmoid(dimage, sigmoid_cache)

		conv3_dx, conv3_dw, conv3_db = bp_conv(sigmoid_dx, conv3_cache)
		grads['W4'] = conv3_dw + self.reg * self.params['W4']
		grads['b4'] = conv3_db

		relu3_dx = bp_relu(conv3_dx, relu3_cache)

		conv2_dx, conv2_dw, conv2_db = bp_conv(relu3_dx, conv2_cache)
		grads['W3'] = conv2_dw + self.reg * self.params['W3']
		grads['b3'] = conv2_db

		relu2_dx = bp_relu(conv2_dx, relu2_cache)

		conv1_dx, conv1_dw, conv1_db = bp_conv(relu2_dx, conv1_cache)
		grads['W2'] = conv1_dw + self.reg * self.params['W2']
		grads['b2'] = conv1_db

		relu1_dx = bp_relu(conv1_dx, relu1_cache)

		fc1_dx, fc1_dw, fc1_db = bp_fc(relu1_dx, fc1_cache)
		grads['W1'] = fc1_dw + self.reg * self.params['W1']
		grads['b1'] = fc1_db

		return loss, grads
"""

