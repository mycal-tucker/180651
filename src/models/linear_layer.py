import keras.backend as K
from keras.activations import softmax
from keras.layers import Layer


class LinearLayer(Layer):
    def __init__(self, num_prototypes, **kwargs):
        self.num_prototypes = num_prototypes
        super(LinearLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.linear_weights = self.add_weight(name='linear_weights',
                                       shape=(self.num_prototypes, 10),  # 10 digits, right?
                                       initializer='uniform',
                                       trainable=True)
        super(LinearLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        return softmax(K.dot(x, self.linear_weights))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 10)
