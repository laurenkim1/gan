
import numpy as np
from solver import *
from model import *
from mnist_loader import *
import pickle
import sys

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = load_data_wrapper()

normalized_data = {
  'X_train': (data['X_train'] - np.mean(data['X_train'], axis = 0)) / np.var(data['X_train']),
  'y_train': data['y_train'],
  'X_val': (data['X_val'] - np.mean(data['X_val'], axis = 0)) / np.var(data['X_val']),
  'y_val': data['y_val'],
  # 'X_test': data['X_test'][:small_set_size],
  # 'y_test': data['y_test'][:small_set_size]
}

"""
model_name = 
model = CNN(weight_scale=0.001, hidden_dim=500, reg=0.001, batch_normalize = True)
solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()
model_out = open(model_name, 'w')
pickle.dump(model, model_out)
loss_history_out = open(model_name[:-4] + "_loss_history.pkl", 'w')
pickle.dump(solver.loss_history, loss_history_out)
model_out.close()
loss_history_out.close()
"""

filenames = ['batchnorm_model_added_bnoption_true.pkl', 'batchnorm_model_added_bnoption_false.pkl']

for inname in filenames:
  infile = open(inname, 'rb')
  model = pickle.load(infile)
  infile.close()
  print inname
  solver = Solver(model, data,
                  num_epochs=1, batch_size=50,
                  update_rule='adam',
                  optim_config={
                    'learning_rate': 1e-4,
                  },
                  verbose=True, print_every=20)


  train_count, train_total = 0.0, 0.0
  print ""
  for image in range(len(data['X_train'])):
    X, y_pred = solver.predict(data['X_train'][image], data['y_train'][image])
    if y_pred == data['y_train'][image]:
      train_count += 1
    sys.stdout.write("\033[F")
    sys.stdout.write(str(image + 1) + " of " + str(len(data['X_train']))+ "\n")
    sys.stdout.flush()
    train_total += 1

  sys.stdout.write("Train Accuracy = %f\n" % (train_count / train_total))
  sys.stdout.flush()

  test_count, test_total = 0.0, 0.0
  print ""
  for image in range(len(data['X_test'])):
    X, y_pred = solver.predict(data['X_test'][image], data['y_test'][image])
    if y_pred == data['y_test'][image]:
      test_count += 1
    test_total += 1
    sys.stdout.write("\033[F")
    sys.stdout.write(str(image + 1) + " of " + str(len(data['X_test'])) + "\n")
    sys.stdout.flush()
  print "Test Accuracy = %f" % (test_count / test_total)