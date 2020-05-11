import os
from utils.metric_tracker_store import MetricTrackerStore
from utils.metric_tracker import MetricTracker
import csv
from utils.plotting import plot_multiple_runs, plot_bar_chart
import numpy as np

# Script reads in the csv files accompanying every trained model and plots some of the values. None of the saved models
# themselves are actually loaded, because that takes a lot of time. If you need to do analysis on the saved models, see
# the load_models.py script.

NUM_DUPLICATES = 10

base_path = '../../../saved_models'
running_path = base_path
store = MetricTrackerStore()  # The idea is to load metrics into this store and then plot them later.
for inv_dir in os.listdir(base_path):
    if not os.path.isdir(base_path + '/' + inv_dir):
        continue
    inverted = 'not' not in inv_dir
    running_path += '/' + inv_dir
    for i_dir in os.listdir(running_path):
        running_path += '/' + i_dir
        identity = 'not' not in i_dir
        for depth_dir in os.listdir(running_path):
            running_path += '/' + depth_dir
            depth = int(depth_dir)
            for num_protos_dir in os.listdir(running_path):
                running_path += '/' + num_protos_dir
                num_protos = int(num_protos_dir)
                for model_id in range(NUM_DUPLICATES):
                    tracker = MetricTracker(depth=depth, num_prototypes=num_protos, model_id=model_id,
                                            identity=identity, inverted=inverted)
                    print()
                    print("Loading")
                    print("Inverted", inverted)
                    print("Identity", identity)
                    print("Depth", depth)
                    print("Num protos", num_protos)
                    print("Trial id", model_id)
                    # Load in the csv
                    with open(running_path + '/' + str(model_id) + '_metrics.csv', 'r') as metrics_file:
                        reader = csv.reader(metrics_file)
                        for line_idx, line in enumerate(reader):
                            print("Reading line", line)
                            if line_idx == 0:
                                svd_cutoff = [int(entry) for entry in line]
                                tracker.record_metrics('svd', svd_cutoff)
                            elif line_idx == 1:  # Reconstruction loss
                                recons_losses = [float(entry) for entry in line]
                                tracker.record_metrics('recons', recons_losses)
                            elif line_idx == 2:  # Classification acc
                                classification_accs = [float(entry) for entry in line]
                                tracker.record_metrics('acc', classification_accs)
                    store.add_tracker(tracker)
                running_path = running_path[:-len(num_protos_dir) - 1]
            running_path = running_path[:-len(depth_dir) - 1]
        running_path = running_path[:-len(i_dir) - 1]
    running_path = running_path[:-len(inv_dir) - 1]

plots_inverted = True
plots_identity = False
# Plot stuff from the store.
# Compare svd cutoffs by depths.
cutoff_means = []
cutoff_stds = []
batch_idxs = []
for depth in range(1, 5):
    cutoffs = store.get_matching_metric_values('svd', req_depth=depth, req_inverted=plots_inverted, req_identity=plots_identity)
    cutoff_means.append(np.mean(cutoffs, axis=0).tolist()[:9000])
    cutoff_stds.append(np.std(cutoffs, axis=0).tolist()[:9000])
    batch_idxs.append([i for i in range(len(cutoffs[0]))][:9000])
plot_multiple_runs(batch_idxs, cutoff_means, y_stdev=cutoff_stds, labels=['Depth ' + str(i + 1) for i in range(len(cutoff_means))], x_axis="Batch idx", y_axis="Cutoff")

# Compare accuracy by depths.
acc_means = []
acc_stds = []
acc_labels = []
for depth in range(1, 5):
    accs = store.get_matching_metric_values('acc', req_depth=depth, req_inverted=plots_inverted, req_identity=plots_identity)
    acc_means.append(np.mean(accs, axis=0).tolist())
    acc_stds.append(np.std(accs, axis=0).tolist())
    acc_labels.append([i for i in range(3)])
plot_bar_chart(acc_labels, acc_means, y_stdev=acc_stds, labels=['Depth ' + str(i + 1) for i in range(len(acc_means))],
               x_axis='Training stage', y_axis='Accuracy')


# Vary num prototypes
num_prototypes = [10, 15, 20, 25]
cutoff_means = []
cutoff_stds = []
batch_idxs = []
for num_protos in num_prototypes:
    cutoffs = store.get_matching_metric_values('svd', req_num_prototypes=num_protos, req_depth=1, req_inverted=plots_inverted, req_identity=plots_identity)
    cutoff_means.append(np.mean(cutoffs, axis=0).tolist()[8000:])
    cutoff_stds.append(np.std(cutoffs, axis=0).tolist()[8000:])
    batch_idxs.append([i for i in range(len(cutoffs[0]))][8000:])
plot_multiple_runs(batch_idxs, cutoff_means, y_stdev=cutoff_stds, labels=['Num protos ' + str(num_prototypes[i]) for i in range(len(cutoff_means))], x_axis="Batch idx", y_axis="Cutoff")

