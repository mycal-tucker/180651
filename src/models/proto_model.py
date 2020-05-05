import keras.backend as K
from keras.layers import Dense, Input, Lambda
from models.proto_layer import ProtoLayer
from models.predictor import Predictor
from keras.models import Model
from keras.losses import mse, categorical_crossentropy
import numpy as np
from utils.plotting import plot_rows_of_images


class ProtoModel:
    def __init__(self, num_prototypes, latent_dim, predictor_depth):
        self.num_prototypes = num_prototypes
        self.latent_dim = latent_dim
        self.predictor_depth = predictor_depth
        self.classification_weight = 10
        self.reconstruction_weight = 1
        self.close_to_proto_weight = 1
        self.proto_close_to_weight = 1

        self.input_layer = Input(shape=(784,))  # Actual MNIST input.
        self.encoder, self.decoder, self.predictor, self.proto_layer, self.latent, self.recons = self.build_parts()

        self.auto = self.build_model()

    # Helper function for putting in lambda layers
    @staticmethod
    def get_min(x):
        return K.min(x, axis=1)

    def build_parts(self):
        # Define all the layers for the functional API
        # Encoder part.
        enc_layer1 = Dense(128, activation='relu')
        enc_layer2 = Dense(128, activation='linear')
        enc_layer3 = Dense(self.latent_dim, activation='linear')  # The latent encoding.
        # Decoder part.
        dec_layer2 = Dense(128, activation='linear')
        dec_layer1 = Dense(128, activation='relu')
        recons_layer = Dense(784, activation='sigmoid', name='Reconstruction')
        # Prototypes part
        proto_layer = ProtoLayer(self.num_prototypes, self.latent_dim)

        # Build encoder into model
        enc1 = enc_layer1(self.input_layer)
        enc2 = enc_layer2(enc1)
        enc_output = enc_layer3(enc2)
        encoder = Model(self.input_layer, enc_output, name="Encoder")
        # Build decoder into model
        dec_input = Input(shape=(self.latent_dim,))
        dec2 = dec_layer2(dec_input)
        dec1 = dec_layer1(dec2)
        dec_recons = recons_layer(dec1)
        decoder = Model(dec_input, dec_recons, name="Decoder")

        # All the parts of the net are built separately.
        latent = encoder(self.input_layer)
        recons = decoder(latent)
        predictor = Predictor(self.num_prototypes, self.predictor_depth, sparse=False)
        return encoder, decoder, predictor, proto_layer, latent, recons

    def build_model(self):
        label_layer = Input(shape=(10,))  # The y labels. Needed as input for custom loss.
        # Distances from latents to prototypes and prototypes to latents.
        proto_distances, feature_distances = self.proto_layer(self.latent)
        min_proto_dist = Lambda(ProtoModel.get_min)(proto_distances)
        min_feature_dist = Lambda(ProtoModel.get_min)(feature_distances)
        prediction = self.predictor.model(proto_distances)
        auto = Model([self.input_layer, label_layer], [self.recons, prediction], "Autoencoder")

        # Define various loss tensors.
        recons_loss = mse(self.input_layer, self.recons)
        pred_loss = categorical_crossentropy(label_layer, prediction)
        proto_dist_loss = K.mean(min_proto_dist)
        feature_dist_loss = K.mean(min_feature_dist)
        overall_loss = self.reconstruction_weight * K.mean(recons_loss) +\
                       self.classification_weight * K.mean(pred_loss) +\
                       self.proto_close_to_weight * proto_dist_loss +\
                       self.close_to_proto_weight * feature_dist_loss
        auto.add_loss(overall_loss)
        auto.compile(optimizer='adam', loss='')
        return auto

    def get_predictor_svd(self, num_components=None):
        predictor_approx = self.predictor.get_svd(num_components=num_components)
        # Create a 1-layer approximation and load in the weights.
        new_predictor = Predictor(self.num_prototypes, 1, sparse=False)
        new_predictor.model.set_weights([predictor_approx])
        self.predictor = new_predictor  # TODO: I do want to look into just creating a separate approximation net...
        self.auto = self.build_model()  # Puts the predictor into this model. Could create new one if prefer...

    def use_fewer_protos(self, new_num_protos=10):
        predictor_approx, protos_to_keep, dropoff_svd = self.predictor.decrease_num_prototypes()
        # Create a 1-layer approximation and load in the weights.
        new_predictor = Predictor(new_num_protos, 1, sparse=False)  # Only 1 layer
        new_predictor.model.set_weights([predictor_approx])
        self.predictor = new_predictor
        new_protos = ProtoLayer(new_num_protos, self.latent_dim)
        current_weights = self.proto_layer.get_weights()
        current_weights[0] = current_weights[0][protos_to_keep, :]
        self.num_prototypes = new_num_protos
        self.proto_layer = new_protos
        self.auto = self.build_model()  # Puts the predictor into this model. Could create new one if prefer...
        new_protos.set_weights(current_weights)
        return protos_to_keep, dropoff_svd

    def train(self, inputs, epochs, batch_size, verbosity=1):
        self.auto.fit(inputs, epochs=epochs, batch_size=batch_size, verbose=verbosity)

    def evaluate(self, x_test, y_test_one_hot, y_test, plot=True):
        # Manually evaluate prediction accuracy
        test_reconstructions, test_predictions = self.auto.predict([x_test, y_test_one_hot])
        num_evaluated = 0
        num_correct = 0
        for i, test_prediction in enumerate(test_predictions):
            digit_prediction = np.argmax(test_prediction)
            num_evaluated += 1
            if digit_prediction == y_test[i]:
                num_correct += 1
        accuracy = num_correct / num_evaluated
        print("Label accuracy:", accuracy)
        # Evaluate reconstruction error
        reconstruction_error = np.mean(np.square(test_reconstructions - x_test))
        print("Reconstruction error:", reconstruction_error)
        if not plot:
            return reconstruction_error, accuracy, None, None

        # Plot and save reconstructions
        NUM_EXAMPLES_TO_PLOT = 10
        img_save_location = '../../../saved_models/'
        originals = np.zeros((NUM_EXAMPLES_TO_PLOT, 784))
        reconstructions = np.zeros((NUM_EXAMPLES_TO_PLOT, 784))
        predictions = []
        for test_idx in range(NUM_EXAMPLES_TO_PLOT):
            test_input = np.reshape(x_test[test_idx], (1, -1))
            # Get the prediction
            reconstruction, prediction = self.auto.predict([test_input, np.reshape(y_test_one_hot[test_idx], (1, -1))])
            originals[test_idx] = test_input
            reconstructions[test_idx] = reconstruction
            predicted_digit = np.argmax(prediction)
            predictions.append(predicted_digit)
        plot_rows_of_images([originals, reconstructions], img_save_location + 'reconstructions')

        # Plot and save the prototypes
        proto_weights = self.proto_layer.get_weights()[0]
        decoded_prototypes = np.zeros((self.num_prototypes, 784))
        for proto_idx in range(self.num_prototypes):
            proto_enc = np.reshape(proto_weights[proto_idx, :], (1, -1))
            # Pass through decoder
            decoded_proto = self.decoder.predict(proto_enc)
            decoded_prototypes[proto_idx] = decoded_proto
        plot_rows_of_images([decoded_prototypes], savepath=img_save_location + 'prototypes')

        return reconstruction_error, accuracy, None, None

    def save_model(self, filepath):
        self.auto.save_weights(filepath + 'auto.h5')
        self.decoder.save_weights(filepath + 'decoder.h5')

    def load_model(self, filepath):
        self.auto.load_weights(filepath + 'auto.h5')
        self.decoder.load_weights(filepath + 'decoder.h5')
