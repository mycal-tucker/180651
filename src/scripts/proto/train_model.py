from data_parsing.mnist_data import get_data
from models.proto_model import ProtoModel


# Hyperparameters
NUM_PROTOS = 25
LATENT_DIM = 10
NUM_EPOCHS = 5
PREDICTOR_DEPTH = 1

auto = ProtoModel(NUM_PROTOS, LATENT_DIM, PREDICTOR_DEPTH)

# Get the MNIST data.
x_train, _, y_train_one_hot, x_test, y_test, y_test_one_hot = get_data()

# Start training. Note how labels are passed in as an input.
auto.train([x_train, y_train_one_hot], epochs=NUM_EPOCHS, batch_size=32)

# Evaluate.
auto.evaluate(x_test, y_test_one_hot, y_test)

# Do some SVD stuff?
# auto.get_predictor_svd()
auto.use_fewer_protos()

auto.evaluate(x_test, y_test_one_hot, y_test)

# Fine-tuning
auto.train([x_train, y_train_one_hot], epochs=1, batch_size=32)
auto.evaluate(x_test, y_test_one_hot, y_test)
