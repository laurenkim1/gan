import numpy as np
from mnist_loader import *
import sys
from sklearn.ensemble import RandomForestClassifier as RFC
import pickle

class NaiveBayes:
	def __init__(self, num_classes, alpha = 1, dimensions = (28,28), use_model = False,
		class_probs = None, conditional_probs = None):
		self.num_classes = num_classes
		self.alpha = alpha
		self.dimensions = dimensions
		if not use_model:
			self.pixel_counts = np.zeros((num_classes, dimensions[0], dimensions[1]))
			self.conditional_probs = np.zeros((num_classes, dimensions[0], dimensions[1]))
			# self.marginal_probs = np.zeros(dimensions)
			self.class_counts = np.zeros(num_classes)
			self.class_probs = np.zeros(num_classes)
			for c in range(num_classes):
				self.class_counts[c] += alpha
				for i in range(dimensions[0]):
					for j in range(dimensions[1]):
						self.pixel_counts[c,i,j] += alpha
		elif class_probs is not None and conditional_probs is not None:
			class_prob_file = open('NB_class_probs.pkl')
			conditional_prob_file = open('NB_conditional_probs.pkl')
			self.class_probs = pickle.load(class_prob_file)
			self.conditional_probs = pickle.load(conditional_prob_file)
		else:
			raise(ValueError("Invalid Intialization of NaiveBayes")) 

	def train(self, X_train, y_train):
		n = len(y_train)
		for d in range(n):
			X = X_train[d, 0]
			y = y_train[d]
			self.class_counts[y] += 1
			for i in range(self.dimensions[0]):
				for j in range(self.dimensions[1]):
					v = round(X[i,j])
					self.pixel_counts[y,i,j] += v
			
		for c in range(self.num_classes):
			self.conditional_probs[c] = self.pixel_counts[c] / self.class_counts[c]
			# self.marginal_probs += self.pixel_counts[c] / n
			self.class_probs[c] = self.class_counts[c] / sum(self.class_counts)

		class_out = open('NB_class_probs.pkl', 'w')
		conditional_out = open('NB_conditional_probs.pkl', 'w')
		pickle.dump(self.conditional_probs, conditional_out)
		pickle.dump(self.class_probs, class_out)


	def predict(self, X, y = None):
		predictions = np.zeros(len(X))
		sys.stdout.write('Predictions Made: \n\n')
		for d, x in enumerate(X):
			probs = np.zeros(self.num_classes)
			# Initalize Marginal probabilities
			for c in range(self.num_classes):
				probs[c] += -np.log(self.class_probs[c])
			
			x = x[0]
			d1, d2 = self.dimensions[0], self.dimensions[1]
			for i in range(d1):
				for j in range(d2):
					for c in range(self.num_classes):
						v = round(x[i,j])
						if v:
							probs[c] += -np.log(self.conditional_probs[c,i,j])
						else:
							probs[c] += -np.log(1 - self.conditional_probs[c,i,j])

			predictions[d] = np.argmin(probs)
			sys.stdout.write("\033[F")
			sys.stdout.write(str(d + 1) + " of " + str(len(X)) + "\n")
			sys.stdout.flush()

		if y is None:
			return predictions
		else:
			count = 0.0
			n = len(y)
			for i in range(n):
				if y[i] == predictions[i]:
					count += 1
			return count / float(n), predictions

data = load_data_wrapper()
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

# NB = NaiveBayes(10)
# print "Training..."
# NB.train(X_train, y_train)
# print "Computing training accuracy..."
# train_accuracy, predictions = NB.predict(X_train, y_train)
# print "Training set accuracy = %f" % train_accuracy
# print "Computing test set accuracy..."
# test_accuracy, predictions = NB.predict(X_test, y_test)
# print "Test set accuracy = %f" % test_accuracy

print "Now loading the pretrained model..."

pretrained_NB = NaiveBayes(10, use_model = True, class_probs = 'NB_class_probs.pkl', 
	conditional_probs = 'NB_conditional_probs.pkl')

train_accuracy, predictions = pretrained_NB.predict(X_train, y_train)
print "Training set accuracy = %f" % train_accuracy

test_accuracy, predictions = pretrained_NB.predict(X_test, y_test)
print "Test set accuracy = %f" % test_accuracy
