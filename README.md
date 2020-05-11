# 180651 Final Project by Mycal Tucker

Codebase for final project for 18.0651 Spring 2020.
Implements the Prototype Case Network by Li et al., and explores several types of modifications.

To get started, install all the dependencies in requirements.txt in a virtual environment running Python 3.6.
Also, create an empty directory called `saved_models` alongside `src` -- your saved models and files will go here.

The basic architecture of this project is as follows:
1) data_parsing holds the files for generating the datasets to use.
In this case, that's two types of MNIST data, living in mnist_data.py
2) models holds the objects for different parts of the network.
proto_model.py defines the ProtoModel, the overall model used for testing.
The other parts - linear_layer, predictor, and proto_layer, define subcomponents.
3) scripts holds the actual executable python scripts that run.
exploration has train_model.py, a script that trains just a single model.
trails holds a few scripts for training and analyzing lots of models.
training_trials.py trains and saves lot of models.
gen_plots and load_models use those trained models for analysis and plots.
4) utils holds a couple utilities that are broadly useful like plotting and metric storing.

That's the whole structure!

To get started, try running train_model.py.
If you see an error about saved_models, make sure you created that directory!

Once you've got that running, you can train a bunch of models using the scripts in training_trials.

Have fun!
