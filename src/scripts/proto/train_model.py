from data_parsing.mnist_data import *
from models.predictor import Predictor
from models.proto_model import ProtoModel
from utils.plotting import plot_multiple_runs

# Hyperparameters
NUM_PROTOS = 10
LATENT_DIM = 10
NUM_EPOCHS = 5
PREDICTOR_DEPTH = 1
Predictor.invert_dist = True
Predictor.identity = False

auto = ProtoModel(NUM_PROTOS, LATENT_DIM, PREDICTOR_DEPTH)

# Get the MNIST data.
# x_train, _, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_digit_data()
x_train, _, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_fashion_data()

# Start training. Note how labels are passed in as an input.
# auto.train([x_train, y_train_one_hot], epochs=NUM_EPOCHS, batch_size=32)
cutoffs = auto.train_with_metrics([x_train, y_train_one_hot], epochs=NUM_EPOCHS, batch_size=128)
plot_multiple_runs([[i for i in range(len(cutoffs))]], [cutoffs], y_stdev=None, labels=[["SVD cutoff"]], x_axis="Epoch idx", y_axis="Cutoff")
auto.viz_latent_space(x_test, y_test, class_labels)

# Evaluate.
auto.evaluate(x_test, y_test_one_hot, y_test)

# Do some SVD stuff?
# auto.get_predictor_svd()
auto.use_fewer_protos()

auto.evaluate(x_test, y_test_one_hot, y_test)

# Fine-tuning
new_cutoffs = auto.train_with_metrics([x_train, y_train_one_hot], epochs=5, batch_size=32)
cutoffs.extend(new_cutoffs)
plot_multiple_runs([[i for i in range(len(cutoffs))]], [cutoffs], y_stdev=None, labels=[["SVD cutoff"]], x_axis="Epoch idx", y_axis="Cutoff")
auto.evaluate(x_test, y_test_one_hot, y_test)
