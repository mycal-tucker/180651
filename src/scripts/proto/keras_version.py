import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.losses import mse, categorical_crossentropy
import keras.backend as K

from data_parsing.mnist_data import get_data

# Create a simple model
inp = Input(shape=(784,))  # Actual MNIST input.
label_layer = Input(shape=(10,))  # The y labels. Needed as input for custom loss.
# Encoder part.
enc1 = Dense(128, activation='relu')(inp)
enc2 = Dense(128, activation='linear')(enc1)
latent = Dense(10, activation='linear')(enc2)  # The latent encoding.
# Decoder part.
dec2 = Dense(128, activation='linear')(latent)
dec1 = Dense(128, activation='relu')(dec2)
recons = Dense(784, activation='sigmoid', name='Reconstruction')(dec1)
# Predictor part.
prediction = Dense(10, activation='softmax', name='Prediction')(latent)

# Assemble into one big model.
auto = Model([inp, label_layer], [recons, prediction], "Autoencoder")
# Define various loss tensors.
recons_loss = mse(inp, recons)
pred_loss = categorical_crossentropy(label_layer, prediction)
overall_loss = K.mean(recons) + K.mean(pred_loss)
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
