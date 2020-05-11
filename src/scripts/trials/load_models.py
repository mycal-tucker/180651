import os
from models.proto_model import ProtoModel
import tensorflow as tf
import keras.backend as K
from data_parsing.mnist_data import *
from models.predictor import Predictor
from utils.metric_tracker_store import MetricTrackerStore
from utils.metric_tracker import MetricTracker
from utils.plotting import plot_multiple_runs, plot_bar_chart

# Choose which dataset you want to run with by toggling this boolean.
digit_task = False
if digit_task:
    base_path = '../../../saved_models'
    x_train, _, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_digit_data()
else:  # Fashion task.
    base_path = '../../../saved_models/fashion'
    x_train, _, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_fashion_data()
running_path = base_path
NUM_DUPLICATES = 10
LATENT_DIM = 10

# Loading models takes time, so if you want to decrease load times, only pull in the types of models you want.
# If you're not sure, you can leave values as None, and all will be loaded.
req_inv = None
req_identity = False
req_num_protos = 10
req_depth = 1

store = MetricTrackerStore()
for inv_dir in os.listdir(base_path):
    if not os.path.isdir(base_path + '/' + inv_dir):
        continue
    if 'fashion' in inv_dir:
        # The directory structure got a bit messed up, so this prevents digit tasks loading fashion models
        continue
    inverted = 'not' not in inv_dir
    if req_inv is not None and req_inv != inverted:
        continue
    Predictor.invert_dist = inverted
    running_path += '/' + inv_dir
    for i_dir in os.listdir(running_path):
        identity = 'not' not in i_dir
        if req_identity is not None and req_identity != identity:
            continue
        Predictor.identity = identity
        running_path += '/' + i_dir
        for depth_dir in os.listdir(running_path):
            depth = int(depth_dir)
            if req_depth is not None and req_depth != depth:
                continue
            running_path += '/' + depth_dir
            for num_protos_dir in os.listdir(running_path):
                num_protos = int(num_protos_dir)
                if req_num_protos is not None and req_num_protos != num_protos:
                    continue
                running_path += '/' + num_protos_dir
                for model_id in range(NUM_DUPLICATES):
                    print()
                    print("Loading")
                    print("Inverted", inverted)
                    print("Identity", identity)
                    print("Depth", depth)
                    print("Num protos", num_protos)
                    print("Trial id", model_id)
                    tracker = MetricTracker(depth=depth, num_prototypes=num_protos, model_id=model_id, inverted=inverted,
                                            identity=identity)
                    # Load the model so we can do stuff with it.
                    auto = ProtoModel(num_protos, LATENT_DIM, depth)
                    auto.load_model(running_path + '/' + str(model_id) + '_original_')  # I want the original one.
                    # If you want to visualize the latent space, uncomment the line below. It pauses execution, though.
                    # auto.viz_latent_space(x_test, y_test, class_labels)
                    # Test it with data.
                    _, original_acc, _, _ = auto.evaluate(x_test, y_test_one_hot, y_test, plot=False)
                    print("Original", original_acc)
                    # Get the SVD values and ratios.
                    sigmas, ratios = auto.get_sigma_values()
                    tracker.record_metrics('sigmas', sigmas.tolist())
                    tracker.record_metrics('ratios', [1.0] + ratios)
                    # Replace it with the SVD approximation. Incrementally decrease the number of components to get
                    # a full range of results.
                    total_falloffs = [0]
                    incremental_falloffs = [0]
                    num_components = [10]
                    prev_acc = original_acc
                    for i in range(9, 0, -1):
                        auto.get_predictor_svd(i)
                        _, approx_acc, _, _ = auto.evaluate(x_test, y_test_one_hot, y_test, plot=False)
                        print("Approximated", approx_acc)
                        incremental = prev_acc - approx_acc
                        prev_acc = approx_acc
                        total = original_acc - approx_acc
                        total_falloffs.append(total)
                        incremental_falloffs.append(incremental)
                        num_components.append(i)

                    tracker.record_metrics('num_components', num_components)
                    tracker.record_metrics('total_approx', total_falloffs)
                    tracker.record_metrics('incremental_approx', incremental_falloffs)
                    store.add_tracker(tracker)
                    tf.reset_default_graph()
                    K.clear_session()
                running_path = running_path[:-len(num_protos_dir) - 1]
            running_path = running_path[:-len(depth_dir) - 1]
        running_path = running_path[:-len(i_dir) - 1]
    running_path = running_path[:-len(inv_dir) - 1]

# Plot loss in performance by stepping through how many SVD components you keep.
comps = []
total_loss = []
total_loss_std = []
rel_loss = []
rel_loss_std = []
labels = []
for inv in [True, False]:
    for iden in [False]:
        comps.append(np.mean(store.get_matching_metric_values('num_components', req_inverted=inv, req_identity=iden), axis=0).tolist())
        total_loss.append(np.mean(store.get_matching_metric_values('total_approx', req_inverted=inv, req_identity=iden), axis=0).tolist())
        total_loss_std.append(np.std(store.get_matching_metric_values('total_approx', req_inverted=inv, req_identity=iden), axis=0).tolist())
        rel_loss.append(np.mean(store.get_matching_metric_values('incremental_approx', req_inverted=inv, req_identity=iden), axis=0).tolist())
        rel_loss_std.append(np.std(store.get_matching_metric_values('incremental_approx', req_inverted=inv, req_identity=iden), axis=0).tolist())
        inv_string = 'Inverted' if inv else "Not Inverted"
        id_string = 'Identity' if iden else 'Not Identity'
        label_string = inv_string + ', ' + id_string
        labels.append(inv_string)
plot_multiple_runs(comps, total_loss, y_stdev=total_loss_std, labels=labels, x_axis="Num components",
                   y_axis="Total loss", window_size=1, top=None, bottom=None)
plot_multiple_runs(comps, rel_loss, y_stdev=rel_loss_std, labels=labels, x_axis="Num components",
                   y_axis="Incremental loss", window_size=1, top=None, bottom=None)

# Plot histogram of sigma values and ratios.
s_vals = []
s_vals_std = []
r_vals = []
r_vals_std = []
bar_labels = []
for inv in [True, False]:
    s_vals.append(np.mean(store.get_matching_metric_values('sigmas', req_inverted=inv, req_identity=False), axis=0).tolist())
    s_vals_std.append(np.std(store.get_matching_metric_values('sigmas', req_inverted=inv, req_identity=False), axis=0).tolist())
    r_vals.append(np.mean(store.get_matching_metric_values('ratios', req_inverted=inv, req_identity=False), axis=0).tolist())
    r_vals_std.append(np.std(store.get_matching_metric_values('ratios', req_inverted=inv, req_identity=False), axis=0).tolist())
    bar_labels.append([i for i in range(10)])
plot_bar_chart(bar_labels, s_vals, y_stdev=s_vals_std, labels=["Inverted", "Not Inverted"], x_axis="Sigma index", y_axis="Sigma Value", top=None, bottom=None)
plot_bar_chart(bar_labels, r_vals, y_stdev=r_vals_std, labels=["Inverted", "Not Inverted"], x_axis="Sigma index", y_axis="Trailing Ratio", top=None, bottom=None)
