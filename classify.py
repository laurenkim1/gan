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

data = load_data_wrapper()

savefile = open('mymodel.pkl', 'rb')
model = pickle.load(savefile)

solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
savefile.close()

for image in range(len(data['X_train'])):
  X, y_pred = solver.predict(data['X_train'][image], data['y_train'][image])
  if y_pred == data['y_train'][image]:
    plt.title('Label is {label}'.format(label=y_pred))
    plt.imshow(X, cmap='gray')
    plt.show()