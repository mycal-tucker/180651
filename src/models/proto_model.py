import keras.backend as K
from keras.layers import Dense, Input, Lambda
from models.linear_layer import LinearLayer
from models.proto_layer import ProtoLayer
from keras.models import Model
from keras.losses import mse, categorical_crossentropy
import numpy as np
from utils.plotting import plot_rows_of_images


class ProtoModel:
    def __init__(self, num_prototypes, latent_dim):
        self.num_prototypes = num_prototypes
        self.latent_dim = latent_dim
        self.classification_weight = 10
        self.reconstruction_weight = 1
        self.close_to_proto_weight = 1
        self.proto_close_to_weight = 1

        self.auto, self.decoder = self.build_model()

    # Helper function for putting in lambda layers
    @staticmethod
    def get_min(x):
        return K.min(x, axis=1)

    def build_model(self):
        # Define all the layers for the functional API
        # Inputs for both the MNIST data but also labels
        input_layer = Input(shape=(784,))  # Actual MNIST input.
        label_layer = Input(shape=(10,))  # The y labels. Needed as input for custom loss.
        # Encoder part.
        enc_layer1 = Dense(128, activation='relu')
        enc_layer2 = Dense(128, activation='linear')
        enc_layer3 = Dense(self.latent_dim, activation='linear')  # The latent encoding.
        # Decoder part.
        dec_layer2 = Dense(128, activation='linear')
        dec_layer1 = Dense(128, activation='relu')
        recons_layer = Dense(784, activation='sigmoid', name='Reconstruction')
        # Prototypes part
        self.proto_layer = ProtoLayer(self.num_prototypes, self.latent_dim)

        # Assemble into one big model and start passing along tensors.
        enc1 = enc_layer1(input_layer)
        enc2 = enc_layer2(enc1)
        latent = enc_layer3(enc2)
        dec2 = dec_layer2(latent)
        dec1 = dec_layer1(dec2)
        recons = recons_layer(dec1)
        # Prototypes part
        # Distances from latents to prototypes and prototypes to latents.
        proto_distances, feature_distances = self.proto_layer(latent)
        min_proto_dist = Lambda(ProtoModel.get_min)(proto_distances)
        min_feature_dist = Lambda(ProtoModel.get_min)(feature_distances)
        # Predictor part.
        prediction = LinearLayer(self.num_prototypes)(proto_distances)

        auto = Model([input_layer, label_layer], [recons, prediction], "Autoencoder")
        # Define various loss tensors.
        recons_loss = mse(input_layer, recons)
        pred_loss = categorical_crossentropy(label_layer, prediction)
        proto_dist_loss = K.mean(min_proto_dist)
        feature_dist_loss = K.mean(min_feature_dist)
        overall_loss = self.reconstruction_weight * K.mean(recons_loss) +\
                       self.classification_weight * K.mean(pred_loss) +\
                       self.proto_close_to_weight * proto_dist_loss +\
                       self.close_to_proto_weight * feature_dist_loss
        auto.add_loss(overall_loss)
        auto.compile(optimizer='adam', loss='')

        # Also build the decoder
        dec_input = Input(shape=(self.latent_dim,))
        proto_dec2 = dec_layer2(dec_input)
        proto_dec1 = dec_layer1(proto_dec2)
        proto_recons = recons_layer(proto_dec1)
        decoder = Model(dec_input, proto_recons, name="Decoder")

        return auto, decoder

    def train(self, inputs, epochs, batch_size):
        self.auto.fit(inputs, epochs=epochs, batch_size=batch_size)

    def evaluate(self, x_test, y_test_one_hot, y_test):
        # losses = self.auto.evaluate([x_test, y_test_one_hot])
        # Manually evaluate prediction accuracy
        test_reconstructions, test_predictions = self.auto.predict([x_test, y_test_one_hot])
        num_evaluated = 0
        num_correct = 0
        for i, test_prediction in enumerate(test_predictions):
            digit_prediction = np.argmax(test_prediction)
            num_evaluated += 1
            if digit_prediction == y_test[i]:
                num_correct += 1
        print("Label accuracy:", num_correct / num_evaluated)

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
