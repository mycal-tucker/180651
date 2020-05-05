from data_parsing.mnist_data import get_data
from models.proto_model import ProtoModel
import numpy as np
import tensorflow as tf
import keras.backend as K

NUM_TRIALS = 3

# Hyperparameters
NUM_PROTOS = 100  # Big parameter to play with.
LATENT_DIM = 10
NUM_EPOCHS = 5
PREDICTOR_DEPTH = 1

all_proto_perf = []
decreased_perf = []
tuned_perf = []
recons_perf = []
num_unique_protos = []
svd_dropoffs = []
for trial_idx in range(NUM_TRIALS):
    x_train, _, y_train_one_hot, x_test, y_test, y_test_one_hot = get_data()
    auto = ProtoModel(NUM_PROTOS, LATENT_DIM, PREDICTOR_DEPTH)
    auto.train([x_train, y_train_one_hot], epochs=NUM_EPOCHS, batch_size=32, verbosity=0)
    recons, acc, _, _ = auto.evaluate(x_test, y_test_one_hot, y_test, plot=False)
    all_proto_perf.append(acc)
    recons_perf.append(recons)
    # Cull
    new_protos, dropoff_svd = auto.use_fewer_protos()
    num_unique_protos.append(len(set(new_protos)))
    svd_dropoffs.append(dropoff_svd)
    _, acc, _, _ = auto.evaluate(x_test, y_test_one_hot, y_test, plot=False)
    decreased_perf.append(acc)
    # Retrain
    auto.train([x_train, y_train_one_hot], epochs=1, batch_size=32, verbosity=0)
    _, acc, _, _ = auto.evaluate(x_test, y_test_one_hot, y_test, plot=False)
    tuned_perf.append(acc)

    tf.reset_default_graph()
    K.clear_session()


print("Reconstruction mean:", np.mean(recons_perf), "std:", np.std(recons_perf))
print("With all, mean:", np.mean(all_proto_perf), "std:", np.std(all_proto_perf))
print("With pruned, mean:", np.mean(decreased_perf), "std:", np.std(decreased_perf))
print("With tuned, mean:", np.mean(tuned_perf), "std:", np.std(tuned_perf))
print()
print("Num protos after prune:", np.mean(num_unique_protos), "std:", np.std(num_unique_protos))
print("Dropoff point for SVD:", np.mean(svd_dropoffs), "std:", np.std(svd_dropoffs))
