# Script loads models and sees how they do compared to their approximations.

import keras.backend as K
import tensorflow as tf
from data_parsing.mnist_data import get_data
from models.proto_model import ProtoModel
from utils.metric_tracker import MetricTracker
from utils.metric_tracker_store import MetricTrackerStore
import numpy as np
from utils.plotting import plot_multiple_runs

depths = [i for i in range(1, 4)]
# num_prototypes = [1, 2, 3, 4, 5, 6, 7, 10]
num_prototypes = [3, 4, 5, 6, 7, 10]
num_duplicate_trials = 3

LATENT_DIM = 10  # TODO: should vary this as well...
NUM_EPOCHS = 10  # And this?

acc_metric_name = 'accuracy'
recons_metric_name = 'reconstruction error'
approx_acc_metric_name = 'approximation accuracy'
approx_acc_loss_metric_name = 'approximation accuracy loss'
metric_store = MetricTrackerStore()
x_train, _, y_train_one_hot, x_test, y_test, y_test_one_hot = get_data()
for depth in depths:
    for num_protos in num_prototypes:
        for trial_idx in range(num_duplicate_trials):
            print("Loading trial number", trial_idx, "depth", depth, "num protos", num_protos)
            # Create the model.
            auto = ProtoModel(num_protos, LATENT_DIM, depth)
            auto.load_model('../../../saved_models/depth' + str(depth) + '_num_protos' + str(num_protos) + "_trial" + str(trial_idx))
            reconstruction_error, accuracy, _, _ = auto.evaluate(x_test, y_test_one_hot, y_test, plot=False)
            # Replace with its approximation
            auto.get_predictor_svd(num_components=3)  # FIXME
            _, approx_accuracy, _, _ = auto.evaluate(x_test, y_test_one_hot, y_test, plot=False)
            # Store the metrics.
            matching_tracker = MetricTracker(depth, num_protos, trial_idx)
            metric_store.add_tracker(matching_tracker)
            matching_tracker.record_metrics(acc_metric_name, accuracy)
            matching_tracker.record_metrics(recons_metric_name, reconstruction_error)
            matching_tracker.record_metrics(approx_acc_metric_name, approx_accuracy )
            matching_tracker.record_metrics(approx_acc_loss_metric_name, accuracy - approx_accuracy)

            tf.reset_default_graph()
            K.clear_session()


# Create plots where depth is the x axis.
def gen_plot_by_proto(metric_name):
    x_data = np.zeros((len(depths), len(num_prototypes)))
    y_data = np.zeros((len(depths), len(num_prototypes)))
    y_stdev_data = np.zeros_like(y_data)
    labels = []
    for depth_idx, depth in enumerate(depths):
        for proto_idx, num_protos in enumerate(num_prototypes):
            metrics = metric_store.get_matching_metric_values(metric_name, req_depth=depth,
                                                                  req_num_prototypes=num_protos)
            mean_val = np.mean(metrics)
            stdev_val = np.std(metrics)
            x_data[depth_idx, proto_idx] = num_protos
            y_data[depth_idx, proto_idx] = mean_val
            y_stdev_data[depth_idx, proto_idx] = stdev_val
        labels.append(str(depth) + " depth")
    plot_multiple_runs(x_data, y_data, y_stdev_data, labels, x_axis="Number of prototypes", y_axis=metric_name)


gen_plot_by_proto(approx_acc_metric_name)
gen_plot_by_proto(approx_acc_loss_metric_name)
