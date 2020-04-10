import keras.backend as K
from keras.activations import softmax
from keras.layers import Layer


class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(LinearLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.linear_weights = self.add_weight(name='linear_weights',
                                              shape=(self.input_dim, self.output_dim),
                                              initializer='uniform',
                                              trainable=True)
        super(LinearLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        return softmax(K.dot(x, self.linear_weights))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
