import numpy as np
import random
import math
import time
from feedforward import * 
from backprop import *
from solver import *
from model import *
from gradient_check import *
from mnist_loader import *
from mnist_loader import *
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

def rel_error(x, y):
	""" returns relative error """
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
"""
model = CNN()

N = 50
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)

loss, grads = model.loss(X, y)
print 'Initial loss (no regularization): ', loss

model.reg = 0.5
loss, grads = model.loss(X, y)
print 'Initial loss (with regularization): ', loss
"""

"""
num_inputs = 2
input_dim = (3, 10, 10)
reg = 0.0
num_classes = 10
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

model = CNN(num_filters=3, filter_size=3,
                          input_dim=input_dim, hidden_dim=7,
                          dtype=np.float64)
loss, grads = model.loss(X, y)
for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
    e = rel_error(param_grad_num, grads[param_name])
    print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))

"""

data = load_data_wrapper()

small_set_size = 5
small_data = {
  'X_train': data['X_train'][:small_set_size],
  'y_train': data['y_train'][:small_set_size],
  'X_val': data['X_val'][:small_set_size],
  'y_val': data['y_val'][:small_set_size],
  # 'X_test': data['X_test'][:small_set_size],
  # 'y_test': data['y_test'][:small_set_size]
}
 
# savefile = open('mymodel.pkl', 'rb')
# model = pickle.load(savefile)
model = CNN(weight_scale=0.001, hidden_dim=500, reg=0.001)


outfile = open('batchnorm_model#####', 'w')

solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()

pickle.dump(solver, outfile)
outfile.close()

train_count, train_total = 0.0, 0.0
for image in range(len(data['X_train'])):
  X, y_pred = solver.predict(data['X_train'][image], data['y_train'][image])
  if y_pred == data['y_train'][image]:
    train_count += 1
    # plt.title('Label is {label}'.format(label=y_pred))
    # plt.imshow(X, cmap='gray')
    # plt.show()
  train_total += 1

print "Train Accuracy = %f" % (train_count / train_total)

test_count, test_total = 0.0, 0.0
for image in range(len(data['X_val'])):
  X, y_pred = solver.predict(data['X_val'][image], data['y_val'][image])
  if y_pred == data['y_val'][image]:
    test_count += 1
  test_total += 1

print "Test Accuracy = %f" % (test_count / test_total)


