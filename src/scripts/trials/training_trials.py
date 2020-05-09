import keras.backend as K
import tensorflow as tf
import os
from data_parsing.mnist_data import *
from models.proto_model import ProtoModel
from models.predictor import Predictor
import csv

LATENT_DIM = 10
NUM_TRAINING_EPOCHS = 20
NUM_TUNING_EPOCHS = 5
NUM_DUPLICATES = 5

depths = [1, 2, 3, 4]
num_prototypes = [10, 15, 20, 25, 50, 100]

for inv in [True, False]:
    inv_str = 'inverted' if inv else 'not_inverted'
    Predictor.invert_dist = inv
    for identity in [True, False]:
        id_str = 'identity' if identity else 'not_identity'
        Predictor.identity = identity
        for depth in depths:
            for num_protos in num_prototypes:
                separator = '/'
                path = separator.join(['../../../saved_models', inv_str, id_str, str(depth), str(num_protos)])
                for duplicate in range(NUM_DUPLICATES):
                    print("Running trial for path", path, "duplicate", duplicate)
                    if os.path.isdir(path):
                        pass
                    else:
                        print("Making it")
                        os.makedirs(path)
                    # Now do the actual training, and then save the model and figures
                    auto = ProtoModel(num_protos, LATENT_DIM, depth)
                    x_train, _, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_digit_data()
                    cutoffs = auto.train_with_metrics([x_train, y_train_one_hot], epochs=NUM_TRAINING_EPOCHS, batch_size=128)
                    original_recons, original_acc, _, _ = auto.evaluate(x_test, y_test_one_hot, y_test, img_save_location=path + '/' + str(duplicate + 5) + '_orig_', plot=True, show=False)
                    auto.save_model(path + '/' + str(duplicate + 5) + '_original_')

                    auto.use_fewer_protos(new_num_protos=min(num_protos, 10))
                    pruned_recons, pruned_acc, _, _ = auto.evaluate(x_test, y_test_one_hot, y_test, img_save_location=path + '/' + str(duplicate + 5) + '_pruned_', plot=True, show=False)
                    auto.save_model(path + '/' + str(duplicate + 5) + '_pruned_')

                    new_cutoffs = auto.train_with_metrics([x_train, y_train_one_hot], epochs=NUM_TUNING_EPOCHS, batch_size=128)
                    cutoffs.extend(new_cutoffs)
                    tuned_recons, tuned_acc, _, _ = auto.evaluate(x_test, y_test_one_hot, y_test, img_save_location=path + '/' + str(duplicate + 5) + '_tuned_', plot=True, show=False)
                    auto.save_model(path + '/' + str(duplicate + 5) + '_tuned_')
                    # Any other metrics I want?
                    with open(path + '/' + str(duplicate + 5) + '_metrics.csv', 'w') as metrics_file:
                        writer = csv.writer(metrics_file)
                        writer.writerow(cutoffs)
                        writer.writerow([original_recons, pruned_recons, tuned_recons])
                        writer.writerow([original_acc, pruned_acc, tuned_acc])

                    tf.reset_default_graph()
                    K.clear_session()
