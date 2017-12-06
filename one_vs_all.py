import numpy as np
from mnist_loader import *
import pickle

def sigmoid(x, weights):
	return (1.0 + np.exp(-(np.dot(x, weights)))) ** -1

class LRClassifier:
	def __init__(self,digit_to_classify, dimensions = (28,28), learning_rate = 0.003, 
		iterations = 10, use_model = False, weight_file = None):
		self.num_weights = dimensions[0] * dimensions[1] + 1
		if not use_model:
			self.weights = np.random.normal(size = self.num_weights)
		elif weight_file is not None:
			infile = open(weight_file, 'r')
			self.weights = pickle.load(infile)
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.digit_to_classify = digit_to_classify

	def train(self, X, y):
		for i in range(self.iterations):
			for xi, yi in zip(X,y):
				y_hat = self.predict(xi)
				if yi == 1:
					self.weights +=   self.learning_rate * xi * (1-y_hat)
				else:
					self.weights += - self.learning_rate * xi * y_hat
		outfile = open('LRC_Weights/LRC_' + str(self.digit_to_classify) + '.pkl', 'w')
		pickle.dump(self.weights, outfile)

	def predict(self, x, y=None):
		return sigmoid(x, self.weights)

data = load_data_wrapper()
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

def flatten_data(X):
	res = []
	for x in X:
		# flatten image and add bias
		flat_x = np.insert(np.ndarray.flatten(x[0]), 0, 1)
		res.append(flat_x)
	return np.array(res)

def get_binary_classes(y, c):
	return np.array(map(lambda yi: 1 if yi == c else -1, y))

def multi_class_predictor(classifiers, flat_X, y):
	correct = 0.0
	n = len(flat_X)
	final_predictions = []
	assert(len(flat_X) == len(y))
	for i in range(n):
		predictions = np.array(map(lambda LRC: LRC.predict(flat_X[i]), classifiers))
		y_hat = np.argmax(predictions)
		final_predictions.append(y_hat)
		if y_hat == y[i]:
			correct += 1.0
	return correct / n, final_predictions


classifiers = [LRClassifier(digit_to_classify = i) for i in range(10)]
flat_X_train = flatten_data(X_train)
flat_X_test = flatten_data(X_test)

# print "Training..."
# for LRC in classifiers:
# 	bin_y = get_binary_classes(y_train, LRC.digit_to_classify)
# 	LRC.train(flat_X_train, bin_y)

# 	count = 0.0
# 	for xi, yi in zip(flat_X_test, bin_y):
# 		pred = LRC.predict(xi)
# 		pred_to_class = 1 if pred > 0.5 else -1
# 		if pred_to_class == yi:
# 			count += 1
# 	print "Accuracy on %d's in MNIST: %.4f" % (LRC.digit_to_classify, count / len(y_test))

# print "Testing..."
# accuracy, final_predictions = multi_class_predictor(classifiers, flat_X_test, y_test)
# print accuracy

print "Testing using pretrained weights"
from_file = [LRClassifier(digit_to_classify = i, use_model = True, 
	weight_file = 'LRC_Weights/LRC_' + str(i) + '.pkl') for i in range(10)]

train_accuracy, train_predictions = multi_class_predictor(from_file, flat_X_train, y_train)
print "Training Accuracy: %f" % train_accuracy

test_accuracy, test_predictions = multi_class_predictor(from_file, flat_X_test, y_test)
print "Test Accuracy: %f" % test_accuracy

