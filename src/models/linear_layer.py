import keras.backend as K
from keras.activations import softmax
from keras.layers import Layer
import numpy as np


class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim, inverse_inputs=False, use_softmax=True, identity=False, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inverse_inputs = inverse_inputs
        self.use_softmax = use_softmax
        self.identity = identity
        super(LinearLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not self.identity:
            self.linear_weights = self.add_weight(name='linear_weights',
                                                  shape=(self.input_dim, self.output_dim),
                                                  initializer='uniform',
                                                  trainable=True)
        else:
            # Set as zeros, values have to be overwritten after calling super.
            self.linear_weights = self.add_weight(name='linear_weights',
                                                  shape=(self.input_dim, self.output_dim),
                                                  initializer='zeros',
                                                  trainable=False)
        super(LinearLayer, self).build(input_shape)
        if self.identity:
            weights = np.zeros((self.input_dim, self.output_dim))
            for i in range(self.input_dim):
                for j in range(self.output_dim):
                    if i % self.output_dim == j:
                        weights[i][j] = 1.0
            self.set_weights([weights])

    def call(self, x, **kwargs):
        if self.inverse_inputs:
            x = 1.0 / x
        product = K.dot(x, self.linear_weights)
        if not self.use_softmax:
            return product
        return softmax(product)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
