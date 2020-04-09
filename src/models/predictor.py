from keras.layers import Dense, Input
from models.linear_layer import LinearLayer
from keras.models import Model
from keras.utils import plot_model


class Predictor:
    def __init__(self, num_prototypes, num_layers):
        self.num_prototypes = num_prototypes
        self.num_layers = num_layers
        self.model = self.build_model()
        plot_model(self.model, to_file='../../../saved_models/predictor.png', show_shapes=True)

    # Last layer is the single linear thing
    def build_model(self):
        input_layer = Input(shape=(self.num_prototypes,))
        dense_counter = 0
        prev_tensor = input_layer
        prev_dim = self.num_prototypes
        while dense_counter < self.num_layers - 1:
            num_units = 128
            temp_dense = Dense(num_units, activation='linear')(prev_tensor)
            prev_tensor = temp_dense
            prev_dim = num_units
            dense_counter += 1
        # Lastly, pass through linear layer
        prediction = LinearLayer(prev_dim)(prev_tensor)
        predictor_model = Model(input_layer, prediction)
        return predictor_model
