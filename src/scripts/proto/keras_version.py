import keras.backend as K
from keras.layers import Dense, Input, Lambda, Layer
from keras.losses import mse, categorical_crossentropy
from keras.models import Model

from data_parsing.mnist_data import get_data

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


# Inputs for both the MNIST data but also labels
inp = Input(shape=(784,))  # Actual MNIST input.
label_layer = Input(shape=(10,))  # The y labels. Needed as input for custom loss.
# Encoder part.
enc1 = Dense(128, activation='relu')(inp)
enc2 = Dense(128, activation='linear')(enc1)
latent = Dense(LATENT_DIM, activation='linear')(enc2)  # The latent encoding.
# Decoder part.
dec2 = Dense(128, activation='linear')(latent)
dec1 = Dense(128, activation='relu')(dec2)
recons = Dense(784, activation='sigmoid', name='Reconstruction')(dec1)
# Prototypes part
# Distances from latents to prototypes and prototypes to latents.
proto_distances, feature_distances = MyLayer()(latent)
min_proto_dist = Lambda(get_min)(proto_distances)
min_feature_dist = Lambda(get_min)(feature_distances)
# Predictor part.
prediction = Dense(10, activation='softmax', name='Prediction')(proto_distances)  # FIXME: even this isn't right because allows too many computations.


# Assemble into one big model.
auto = Model([inp, label_layer], [recons, prediction], "Autoencoder")
# Define various loss tensors.
recons_loss = mse(inp, recons)
pred_loss = categorical_crossentropy(label_layer, prediction)
proto_dist_loss = K.mean(min_proto_dist)
feature_dist_loss = K.mean(min_proto_dist)
overall_loss = reconstruction_weight * K.mean(recons) + classification_weight * K.mean(pred_loss) +\
               proto_close_to_weight * proto_dist_loss + close_to_proto_weight * feature_dist_loss
auto.add_loss(overall_loss)
auto.compile(optimizer='adam', loss='')

# Get the MNIST data.
x_train, _, y_train_one_hot, x_test, _, y_test_one_hot = get_data()

# Start training. Note how labels are passed in as an input.
auto.fit([x_train, y_train_one_hot], epochs=1, batch_size=32)

# Evaluate.
losses = auto.evaluate([x_test, y_test_one_hot], None)
# TODO: actually get useful metrics out now that loss is one giant heap.
print()
print("Losses: " + str(losses))
if len(auto.metrics_names) > 1:
    for i, metric_name in enumerate(auto.metrics_names):
        print(str(metric_name) + ": " + str(losses[i]))
