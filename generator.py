class generator():
	def __init__(self, z_dim=(100), num_filters=32, filter_size=3,
                 hidden_dim=500, num_classes=10, weight_scale=1e-3, reg=1e-3,
                 dtype=np.float32):

		self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.z_dim = z_dim

        self.K = num_filters
        self.F = filter_size
        self.S = 2
        self.P = (filter_size - 1)/2

        # FC 1
        self.params['W1'] = np.random.normal(0, weight_scale, (3136, z_dim))
        self.params['b1'] = np.zeros(3136)

        # conv layer 1: generate 50 features
        self.params['W2'] = np.random.normal(0, weight_scale, (z_dim/2, 1, 3, 3))
        self.params['b2'] = np.zeros(z_dim/2)

        # conv layer 2: generate 25 features
        self.params['W3'] = np.random.normal(0, weight_scale, (z_dim/4, z_dim/2, 3, 3))
        self.params['b3'] = np.zeros(z_dim/4)

        # conv layer 3: final convolution with one output channel
        self.params['W4'] = np.random.normal(0, weight_scale, (1, z_dim/4, 1, 1))
        self.params['b4'] = np.zeros(1)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        # Evaluate loss and gradient for the three-layer convolutional network.
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # feed forward

        # fc layer 1
        fc_cache, fc_out = fc_forward(X, W1, b1)
        relu1_cache, relu1_out = relu_forward(fc_out)
        relu1_out.reshape(1, 56, 56, -1)

        # conv->relu layer 1
        conv1_cache, conv1_out = conv_forward(relu1_out, W2, b2, self.F, self.S, z_dim/2, self.P)
        relu2_cache, relu2_out = relu_forward(conv1_out)
        relu2_out.reshape(56, 56)
        # conv->relu layer 2
        conv2_cache, conv2_out = conv_forward(relu2_out, W3, b3, self.F, self.S, z_dim/4, self.P)
        relu3_cache, relu3_out = relu_forward(conv3_out)
        relu3_out.reshape(56, 56)
        # conv->sigmoid layer 3
        conv3_cache, conv3_out = conv_forward(relu2_out, W4, b4, self.F, self.S, 1, self.P)
        sigmoid_cache, sigmoid_out = relu_forward(conv2_out)


        image = sigmoid_out

        if y is None:
          return image
        
        loss, grads = 0, {}

        # backpropagation

        loss, dscores = softmax_loss(scores, y)
        loss += L2_regularize(self.reg, W1)
        loss += L2_regularize(self.reg, W2)
        loss += L2_regularize(self.reg, W3)
        loss += L2_regularize(self.reg, W4)

        fc2_dx, fc2_dw, fc2_db = bp_fc(dscores, fc2_cache)
        grads['W4'] = fc2_dw + self.reg * self.params['W4']
        grads['b4'] = fc2_db

        relu3_dx = bp_relu(fc2_dx, relu3_cache)

        fc1_dx, fc1_dw, fc1_db = bp_fc(relu3_dx, fc1_cache)
        grads['W3'] = fc1_dw + self.reg * self.params['W3']
        grads['b3'] = fc1_db

        pool2_dx = bp_pool(fc1_dx, pool2_cache)
        relu2_dx = bp_relu(pool2_dx, relu2_cache)

        conv2_dx, conv2_dw, conv2_db = bp_conv(relu2_dx, conv2_cache)
        grads['W2'] = conv2_dw + self.reg * self.params['W2']
        grads['b2'] = conv2_db

        pool1_dx = bp_pool(conv2_dx, pool1_cache)
        relu1_dx = bp_relu(pool1_dx, relu1_cache)

        conv1_dx, conv1_dw, conv1_db = bp_conv(relu1_dx, conv1_cache)
        grads['W1'] = conv1_dw + self.reg * self.params['W1']
        grads['b1'] = conv1_db

        return loss, grads



