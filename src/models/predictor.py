import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model

from models.linear_layer import LinearLayer


class Predictor:
    def __init__(self, num_prototypes, num_layers):
        self.num_prototypes = num_prototypes
        self.num_layers = num_layers
        self.num_classes = 10  # Predict which of 10 digits it is
        self.layers = []
        self.model = self.build_model()
        plot_model(self.model, to_file='../../../saved_models/predictor.png', show_shapes=True)

    # Last layer is the single linear thing.
    # Right now, hidden layers are linear too, but could make Dense or whatever.
    def build_model(self):
        input_layer = Input(shape=(self.num_prototypes,))
        dense_counter = 0
        prev_tensor = input_layer
        prev_dim = self.num_prototypes
        while dense_counter < self.num_layers - 1:
            num_units = 128
            hidden_layer = LinearLayer(prev_dim, num_units, use_softmax=False)
            # hidden_layer = Dense(num_units, use_bias=True, activation='linear')
            self.layers.append(hidden_layer)
            prev_tensor = hidden_layer(prev_tensor)
            prev_dim = num_units
            dense_counter += 1
        # Lastly, pass through linear layer
        last_layer = LinearLayer(prev_dim, self.num_classes)
        self.layers.append(last_layer)
        prediction = last_layer(prev_tensor)
        predictor_model = Model(input_layer, prediction)
        return predictor_model

    def get_svd(self, cutoff_idx=None):
        single_matrix_version = self.get_single_matrix_predictor()
        approximation = self.compute_svd_approx(single_matrix_version, cutoff_idx)
        return approximation

    def get_single_matrix_predictor(self):
        # Take the product of all the linear layers to get a single matrix operation.
        running_product = np.identity(self.num_prototypes)
        for layer in self.layers:
            layer_weights = layer.get_weights()
            assert len(layer_weights) == 1, "Only support linear layers."  # TODO: look into how to support biases?
            running_product = np.matmul(running_product, layer_weights[0])
        return running_product

    def compute_svd_approx(self, matrix, cutoff_idx=None):
        # Take the SVD to see if one can reduce the rank of the predictor matrix.
        u, s, vh = np.linalg.svd(matrix)
        # Plot the sigma values if you'd like
        plt.bar([i for i in range(len(s))], s)
        plt.show()
        ratios = [s[i] / s[i + 1] for i in range(len(s) - 1)]
        plt.bar([i for i in range(len(ratios))], ratios)
        plt.show()

        if cutoff_idx is not None:
            assert cutoff_idx < self.num_classes, "Can't ask for more sigmas than classes to predict."
            pass
        else:
            cutoff_idx = None
            for i in range(len(s) - 1):
                curr_s = s[i]
                next_s = s[i + 1]
                ratio = next_s / curr_s
                if ratio < 0.5:  # This is just totally made up. FIXME
                    cutoff_idx = i
                    break
            if cutoff_idx is None:
                print("Didn't find good cutoff, using all svd components")
                cutoff_idx = -1
            print("Using cutoff idx", cutoff_idx)
        smaller_dim = cutoff_idx + 1
        approximation = np.dot(u[:, :smaller_dim] * s[:smaller_dim], vh[:smaller_dim, :])
        return approximation
