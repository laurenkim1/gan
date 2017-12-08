# backpropagation functions
import numpy as np
from im2col import *

def bp_conv(delta, cache):
	X, W, b, S, P, X_col = cache
	K, depth, F, F = W.shape

	db = np.sum(delta, axis=(0, 2, 3))
	delta = delta.transpose(1,2,3,0).reshape(K,-1)
	dw = delta.dot(X_col.T).reshape(W.shape)
	K, D, F, F = W.shape
	dx_cols = W.reshape(K, -1).T.dot(delta)
	dx = col2im_indices(dx_cols, X.shape, F, F, P, S)

	return dx, dw, db

def bp_fc(delta, cache):
	W, h = cache
	N = h.shape[0]
	h_r = h.reshape(N,-1)
	dW = np.dot(h_r.T, delta)
	db = np.sum(delta, axis=0)
	dX = np.dot(delta, W.T).reshape(h.shape)
	return dX, dW, db

def bp_relu(delta, cache):
	if_gt = (cache > 0)
	dX = delta * if_gt
	return dX

def bp_pool(delta, cache):
	X, F, S = cache
	n_x, d_x, h_x, w_x = X.shape

	out_h = (h_x - F) / S + 1
	out_w = (w_x - F) / S + 1

	dx = np.zeros((n_x, d_x, h_x, w_x))

	for n in range(n_x):
	    for d in range(d_x):
	        for h in range(out_h):
	            for w in range(out_w):
	                pool = X[n, d, h*S:h*S+F, w*S:w*S+F]
	                pool_max = np.amax(pool)
	                if_max = (pool == pool_max)
	                dx[n, d, h*S:h*S+F, w*S:w*S+F] += delta[n,d,h,w] * if_max
	return dx

def bp_batchnorm(delta, cache):
	gamma, beta, mu, var, eps, X_norm = cache

	dbeta = np.sum(delta, axis = 0)
	dgamma = np.sum(delta * X_norm, axis = 0)
	N = float(delta.shape[0])

	a1 = delta * gamma /np.sqrt(var + eps)
	b11 = - np.sum(delta * gamma * mu, axis=0) /np.sqrt(var+eps) /(var + eps)
	b12 = mu * np.ones(delta.shape) / N
	a2 = b11 * b12

	dmu = -1 * np.sum(a1 + a2, axis = 0)
	dx = a1 + a2 + dmu / N * np.ones(delta.shape)

	return dx, dgamma, dbeta




