
import numpy as np
import random
import math
import time
import scipy
from scipy import ndimage, misc
import matplotlib.pyplot as plt


class Model:

    def __init__(self, W, layers):

    	self.W = W
        self.layers = layers
        self.layer_cache = []
        self.layer_out_cache = []
    	self.layer_weight_shapes = []
        self.layer_biases_shapes = []

    def feedforward(self, image):
        prev_out = image

        # forward pass
        for layer in self.layers:
            X = prev_out
            W = prev_out.shape

            if layer == "conv":
                V, weights, bias, out = conv_forward(X, W, 3, 1, 1, 1)
                for i in range(out.shape[0]):
                    plt.imsave('images/cat_conv%d.jpg'%i, out[i])
                for i in range(weights.shape[0]):
                    plt.imsave('images/filter_conv%s.jpg'%i, weights[i].reshape((5,5)))
            elif layer == "pool":
                V, out = max_pool_forward(X, W, 2, 2)
                for i in range(out.shape[0]):
                    plt.imsave('images/pool_pic%s.jpg'%i, out[i])
            elif layer == "fc":
                h, w, d = W
                V, weights, bias, out = affine_forward(X, W, h*w*d)
            elif layer == "final":
                # 2 classes
                V, weights, bias, out = affine_forward(X, W, 2)
            else:
                raise NotImplementedError

            self.layer_cache.append(V)
            self.layer_out_cache.append(out)
            if layer != 'pool':
                self.layer_weight_shapes.append(weights.shape)
                self.layer_biases_shapes.append(bias.shape)
            prev_out = out

        final_activation = prev_activation
        return final_activation

    def backpropagate(self, image, label):
        num_layers = len(self.layers)
        nabla_w = [np.zeros(s) for s in self.layer_weight_shapes]
        nabla_b = [np.zeros(s) for s in self.layer_biases_shapes]

        # set first params on the final layer
        final_output = self.layer_out_cache[-1]
        last_delta = (final_output - label)
        last_weights = None 
        final = True

        # go backwards through the layers of neural net
        for l in range(num_layers - 1, -1, -1):
            inner_layer = l - 1
            if (l-1) <0:
                inner_layer_ix = 0
            outer_layer = l

            layer = self.layers[outer_layer_ix]
            activation = self.layers[inner_layer_ix].output if inner_layer_ix >= 0 else image

            transition = self._get_layer_transition(inner_layer, outer_layer)
            # either input to FC or pool to FC -> going from 3d matrix to 1d
            if transition == '3d_to_1d': # final to fc, fc to fc
                db, dw, last_delta = ()
                final = False

            elif transition == "1d_to_1d":
                if l == 0:
                    activation = image
                # calc delta on the first final layer
                db, dw, last_delta = ()

            elif transition == 'conv_to_pool'
                last_delta = ()

            # going from 3d to 3d matrix -> either input to conv or conv to conv
            elif transition == 'to_conv':
                activation = image 
                last_weights = layer.weights
                db, dw = ()

            else:
                pass

            if transition != 'conv_to_pool':
                # print 'nablasb, db,nabldw, dw, DELTA', nabla_b[inner_layer_ix].shape, db.shape, nabla_w[inner_layer_ix].shape, dw.shape, last_delta.shape
                nabla_b[inner_layer_ix], nabla_w[inner_layer_ix] = db, dw
                last_weights = layer.weights

        return self.layer_out_cache[-1], nabla_b, nabla_w

    def _get_layer_transition(self, inner_ix, outer_ix):
        inner, outer = self.layers[inner_ix], self.layers[outer_ix]
        # either input to FC or pool to FC -> going from 3d matrix to 1d
        if (
            (inner_ix < 0 or isinstance(inner, PoolingLayer)) and 
            isinstance(outer, FullyConnectedLayer)
            ):
            return '3d_to_1d'
        # going from 3d to 3d matrix -> either input to conv or conv to conv
        if (
            (inner_ix < 0 or isinstance(inner, ConvLayer)) and 
            isinstance(outer, ConvLayer)
            ):
            return 'to_conv'
        if (
            isinstance(inner, FullyConnectedLayer) and
            (isinstance(outer, ClassifyLayer) or isinstance(outer, FullyConnectedLayer))
            ):
            return '1d_to_1d'
        if (
            isinstance(inner, ConvLayer) and
            isinstance(outer, PoolingLayer)
            ):
            return 'conv_to_pool'

        raise NotImplementedError

    def gradient_descent(self, training_data, batch_size, eta, num_epochs, lmbda=None, test_data = None):
        training_size = len(training_data)
        if test_data: 
            n_test = len(test_data)

        mean_error = []
        correct_res = []

        for epoch in xrange(num_epochs):
            print "Starting epochs"
            start = time.time()
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k in xrange(0, training_size, batch_size)]
            losses = 0

            for batch in batches:
                loss = self.update_mini_batch(batch, eta)
                losses+=loss
            mean_error.append(round(losses/batch_size,2))
            print mean_error

            if test_data:
                print "################## VALIDATE #################"
                res = self.validate(test_data)
                correct_res.append(res)
                print "Epoch {0}: {1} / {2}".format(
                    epoch, self.validate(test_data), n_test)
                print "Epoch {0} complete".format(epoch)
                # time
                timer = time.time() - start
                print "Estimated time: ", timer
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(correct_res)
        plt.show()

    def update_mini_batch(self, batch, eta):
        nabla_w = [np.zeros(s) for s in self.layer_weight_shapes]
        nabla_b = [np.zeros(s) for s in self.layer_biases_shapes]

        batch_size = len(batch)

        for image, label in batch:
            image = image.reshape((1,28,28))
            _ = self.feedforward(image)
            final_res, delta_b, delta_w = self.backprop(image, label)

            nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]

        ################## print LOSS ############
        error = loss(label, final_res)
        
        num =0
        weight_index = []
        for layer in self.layers:
            if not isinstance(layer,PoolingLayer):
                weight_index.append(num)
            num+=1

        for ix, (layer_nabla_w, layer_nabla_b) in enumerate(zip(nabla_w, nabla_b)):
            layer = self.layers[weight_index[ix]]
            layer.weights -= eta * layer_nabla_w / batch_size
            layer.biases -= eta * layer_nabla_b / batch_size
        return error

    def validate(self,data):
        data = [(im.reshape((1,28,28)),y) for im,y in data]
        test_results = [(np.argmax(self.feedforward(x)),y) for x, y in data]
        return sum(int(x == y) for x, y in test_results)  





