import pickle
import matplotlib.pyplot as plt
import numpy as np
in1 = open('loss_history_no_bn.pkl', 'rb')
loss_history_no_bn = pickle.load(in1)
in2 = open('loss_history_with_bn.pkl', 'rb')
loss_history_with_bn = pickle.load(in2)
in1.close()
in2.close()

spacing = 5
loss_history_no_bn = loss_history_no_bn[::spacing]
loss_history_with_bn = loss_history_with_bn[::spacing]
xs = np.arange(len(loss_history_no_bn)) * spacing

no_bn_line, = plt.plot(xs, loss_history_no_bn)
with_bn_line, = plt.plot(xs, loss_history_with_bn)
plt.legend((no_bn_line, with_bn_line), ('Loss With No BN', 'Loss With BN'))
plt.xlabel('Iterations')
plt.ylabel('loss')
plt.show()