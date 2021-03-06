import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model

from models.linear_layer import LinearLayer


class Predictor:
    invert_dist = False
    identity = False

    def __init__(self, num_prototypes, num_layers):
        self.num_prototypes = num_prototypes
        self.num_layers = num_layers
        self.num_classes = 10  # Predict which of 10 digits it is
        self.layers = []
        self.model = self.build_model()
        plot_model(self.model, to_file='../../../saved_models/predictor.png', show_shapes=True)

    def build_model(self):
        input_layer = Input(shape=(self.num_prototypes,))
        dense_counter = 0
        prev_tensor = input_layer
        prev_dim = self.num_prototypes
        # Only invert the first layer!
        already_inverted = not Predictor.invert_dist
        while dense_counter < self.num_layers - 1:
            num_units = 32
            hidden_layer = LinearLayer(prev_dim, num_units, inverse_inputs=not already_inverted, use_softmax=False)
            self.layers.append(hidden_layer)
            prev_tensor = hidden_layer(prev_tensor)
            prev_dim = num_units
            dense_counter += 1
            already_inverted = True
        # Lastly, pass through linear layer
        last_layer = LinearLayer(prev_dim, self.num_classes, inverse_inputs=not already_inverted, identity=Predictor.identity)
        self.layers.append(last_layer)
        prediction = last_layer(prev_tensor)
        predictor_model = Model(input_layer, prediction)
        return predictor_model

    def get_svd(self, num_components=None):
        single_matrix_version = self.get_single_matrix_predictor()
        approximation = self.compute_svd_approx(single_matrix_version, num_components)
        return approximation

    def get_sigmas(self):
        matrix = self.get_single_matrix_predictor()
        u, s, vh = np.linalg.svd(matrix)
        ratios = [s[i] / s[i + 1] for i in range(len(s) - 1)]
        return s, ratios

    def get_single_matrix_predictor(self):
        # Take the product of all the linear layers to get a single matrix operation.
        running_product = np.identity(self.num_prototypes)
        for layer in self.layers:
            layer_weights = layer.get_weights()
            running_product = np.matmul(running_product, layer_weights[0])
        return running_product

    def compute_svd_approx(self, matrix, num_components=None):
        # Take the SVD to see if one can reduce the rank of the predictor matrix.
        u, s, vh = np.linalg.svd(matrix)
        # Plot the sigma values if you'd like
        # plt.bar([i for i in range(len(s))], s)
        # plt.show()
        # ratios = [s[i] / s[i + 1] for i in range(len(s) - 1)]
        # plt.bar([i for i in range(len(ratios))], ratios)
        # plt.show()

        if num_components is not None:
            assert num_components <= self.num_classes, "Can't ask for more sigmas than classes to predict."
        else:
            num_components = self.find_svd_dropoff(cutoff_ratio=0.5)
            if num_components is None:
                print("Didn't find good cutoff, using all svd components")
                num_components = len(s)
            print("Using cutoff idx", num_components)
        approximation = np.dot(u[:, :num_components] * s[:num_components], vh[:num_components, :])
        return approximation

    def find_svd_dropoff(self, cutoff_ratio=0.5):
        matrix = self.get_single_matrix_predictor()
        u, s, vh = np.linalg.svd(matrix)
        for i in range(len(s) - 1):
            curr_s = s[i]
            next_s = s[i + 1]
            ratio = next_s / curr_s
            if ratio < cutoff_ratio:
                return i + 1
        return len(s)

    def decrease_num_prototypes(self):
        matrix = self.get_single_matrix_predictor()
        dropoff_svd = self.find_svd_dropoff()

        # Just try to directly take best rows of the matrix.
        best_prototypes = []
        for col_id in range(self.num_classes):
            best_prototypes.append(np.argmax(matrix[:, col_id]))
        approximation = matrix[best_prototypes, :self.num_classes]
        return approximation, best_prototypes, dropoff_svd
