import numpy as np
from mnist_loader import *
import sys
from sklearn.ensemble import RandomForestClassifier as RFC

data = load_data_wrapper()
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

def flatten_data(X):
	res = []
	for x in X:
		res.append(np.ndarray.flatten(x[0]))
	return np.array(res)

RF = RFC(n_estimators = 10)
RF.fit(flatten_data(X_train), y_train)
print "Train accuracy: %f" % RF.score(flatten_data(X_train), y_train)
print "Test accuracy: %f" % RF.score(flatten_data(X_test), y_test)
