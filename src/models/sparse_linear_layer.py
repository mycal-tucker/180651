import keras.backend as K
from keras.activations import softmax
from keras.layers import Layer


class SparseLinearLayer(Layer):
    def __init__(self, input_dim, output_dim, use_softmax=True, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_softmax = use_softmax
        super(SparseLinearLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.linear_weights = self.add_weight(name='linear_weights',
                                              shape=(self.input_dim, self.output_dim),
                                              initializer='uniform',
                                              trainable=True)
        self.sparsity_weights = self.add_weight(name='sparsity_weights',
                                                shape=(self.input_dim, self.output_dim),
                                                initializer='uniform',
                                                trainable=True)
        super(SparseLinearLayer, self).build(input_shape)

    # Play around with different costs, etc. Doesn't seem to actually help.
    def call(self, x, **kwargs):
        x = 1.0 / x
        product = K.dot(x, self.linear_weights)
        # Calculate the sparsity loss.
        # Each prototype corresponds to only one class
        for col_idx in range(self.output_dim):
            softmaxed_sparsity = softmax(self.sparsity_weights[:, col_idx: col_idx + 1])
            # curr_lin_weights = K.pow(self.linear_weights[:, col_idx: col_idx + 1], 2)
            curr_lin_weights = K.pow(K.abs(self.linear_weights[:, col_idx: col_idx + 1]), 0.1)
            # curr_lin_weights = self.linear_weights[:, col_idx: col_idx + 1]  # Negative weights are fine
            dotted = K.sum(K.dot(curr_lin_weights, 1 - K.transpose(softmaxed_sparsity)))
            self.add_loss(1 * dotted)
        # Each class corresponds to only one prototype.
        # for row_idx in range(self.input_dim):
        #     softmaxed_sparsity = softmax(self.sparsity_weights[row_idx: row_idx + 1, :])
        #     # curr_lin_weights = K.pow(K.abs(self.linear_weights[row_idx: row_idx + 1, :]), 0.1)
        #     # curr_lin_weights = K.abs(self.linear_weights[row_idx: row_idx + 1, :])
        #     curr_lin_weights = self.linear_weights[row_idx: row_idx + 1, :]
        #     dotted = K.sum(K.dot(K.transpose(curr_lin_weights), 1 - softmaxed_sparsity))
        #     self.add_loss(.1 * dotted)
        # self.add_loss(1 * K.sum(K.pow(K.abs(self.linear_weights), 2)))  # Keep all reasonably small
        if not self.use_softmax:
            return product
        return softmax(product)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
