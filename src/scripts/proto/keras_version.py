import keras.backend as K
import numpy as np
from keras.layers import Dense, Input, Lambda, Layer
from keras.losses import mse, categorical_crossentropy
from keras.models import Model

from data_parsing.mnist_data import get_data
from utils.plotting import plot_rows_of_images

# Hyperparameters
NUM_PROTOS = 15
LATENT_DIM = 10

classification_weight = 10
reconstruction_weight = 1
close_to_proto_weight = 1
proto_close_to_weight = 1


class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.prototypes = self.add_weight(name='proto_kern',
                                          shape=(NUM_PROTOS, LATENT_DIM),
                                          initializer='uniform',
                                          trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        # Compute the distance between x and the protos
        x_squared = K.reshape(MyLayer.get_norms(x), shape=(-1, 1))
        protos_squared = K.reshape(MyLayer.get_norms(self.prototypes), shape=(1, -1))

        dists_to_protos = x_squared + protos_squared - 2 * K.dot(x, K.transpose(self.prototypes))
        dists_to_latents = protos_squared + x_squared - 2 * K.dot(self.prototypes, K.transpose(x))
        # TODO: how do these shapes actually work?
        return [dists_to_protos, dists_to_latents]

    @staticmethod
    def get_norms(x):
        return K.sum(K.pow(x, 2), axis=1)

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], NUM_PROTOS), (NUM_PROTOS, input_shape[0])]


# Helper function for putting in lambda layers
def get_min(x):
    return K.min(x, axis=1)


# Define all the layers for the functional API
# Inputs for both the MNIST data but also labels
input_layer = Input(shape=(784,))  # Actual MNIST input.
label_layer = Input(shape=(10,))  # The y labels. Needed as input for custom loss.
# Encoder part.
enc_layer1 = Dense(128, activation='relu')
enc_layer2 = Dense(128, activation='linear')
enc_layer3 = Dense(LATENT_DIM, activation='linear')  # The latent encoding.
# Decoder part.
dec_layer2 = Dense(128, activation='linear')
dec_layer1 = Dense(128, activation='relu')
recons_layer = Dense(784, activation='sigmoid', name='Reconstruction')
# Prototypes part
proto_layer = MyLayer()

# Assemble into one big model and start passing along tensors.
enc1 = enc_layer1(input_layer)
enc2 = enc_layer2(enc1)
latent = enc_layer3(enc2)
dec2 = dec_layer2(latent)
dec1 = dec_layer1(dec2)
recons = recons_layer(dec1)
# Prototypes part
# Distances from latents to prototypes and prototypes to latents.
proto_distances, feature_distances = proto_layer(latent)
min_proto_dist = Lambda(get_min)(proto_distances)
min_feature_dist = Lambda(get_min)(feature_distances)
# Predictor part.
prediction = Dense(10, activation='softmax', name='Prediction')(proto_distances)  # FIXME: even this isn't right because allows too many computations.

auto = Model([input_layer, label_layer], [recons, prediction], "Autoencoder")
# Define various loss tensors.
recons_loss = mse(input_layer, recons)
pred_loss = categorical_crossentropy(label_layer, prediction)
proto_dist_loss = K.mean(min_proto_dist)
feature_dist_loss = K.mean(min_proto_dist)
overall_loss = reconstruction_weight * K.mean(recons_loss) + classification_weight * K.mean(pred_loss) +\
               proto_close_to_weight * proto_dist_loss + close_to_proto_weight * feature_dist_loss
auto.add_loss(overall_loss)
auto.compile(optimizer='adam', loss='')

# Get the MNIST data.
x_train, _, y_train_one_hot, x_test, y_test, y_test_one_hot = get_data()

# Start training. Note how labels are passed in as an input.
auto.fit([x_train, y_train_one_hot], epochs=1, batch_size=32)

# Evaluate.
losses = auto.evaluate([x_test, y_test_one_hot], None)
# Manually evaluate prediction accuracy
test_reconstructions, test_predictions = auto.predict([x_test, y_test_one_hot])
num_evaluated = 0
num_correct = 0
for i, test_prediction in enumerate(test_predictions):
    digit_prediction = np.argmax(test_prediction)
    num_evaluated += 1
    if digit_prediction == y_test[i]:
        num_correct += 1
print("Label accuracy:", num_correct / num_evaluated)

# Plot reconstructions.
NUM_EXAMPLES_TO_PLOT = 10
img_save_location = '../../../saved_models/'
originals = np.zeros((NUM_EXAMPLES_TO_PLOT, 784))
reconstructions = np.zeros((NUM_EXAMPLES_TO_PLOT, 784))
predictions = []
for test_idx in range(NUM_EXAMPLES_TO_PLOT):
    test_input = np.reshape(x_test[test_idx], (1, -1))
    test_digit = y_test[test_idx]
    # Get the prediction
    reconstruction, prediction = auto.predict([test_input, np.reshape(y_test_one_hot[test_idx], (1, -1))])
    originals[test_idx] = test_input
    reconstructions[test_idx] = reconstruction
    predicted_digit = np.argmax(prediction)
    predictions.append(predicted_digit)
plot_rows_of_images([originals, reconstructions], img_save_location + 'reconstructions')

# Plot the prototypes.
# To do that, I need a decoder model...
dec_input = Input(shape=(LATENT_DIM,))
proto_dec2 = dec_layer2(dec_input)
proto_dec1 = dec_layer1(proto_dec2)
proto_recons = recons_layer(proto_dec1)
decoder = Model(dec_input, proto_recons, name="Decoder")
proto_weights = proto_layer.get_weights()[0]
decoded_prototypes = np.zeros((NUM_PROTOS, 784))
for proto_idx in range(NUM_PROTOS):
    proto_enc = np.reshape(proto_weights[proto_idx, :], (1, -1))
    # Pass through decoder
    decoded_proto = decoder.predict(proto_enc)
    decoded_prototypes[proto_idx] = decoded_proto
plot_rows_of_images([decoded_prototypes], savepath=img_save_location + 'prototypes')