# Neural Architecture Search (NAS) with PyTorch

This project implements a Neural Architecture Search (NAS) algorithm using evolutionary strategies to optimize neural network architectures for different datasets. The project leverages PyTorch for model building and training, and Weights & Biases (wandb) for experiment tracking.


## Experiments

### Neural Architecture search
https://wandb.ai/wsniady/NAS?nw=nwuserwsniady

### Hyperpameter tuning
https://wandb.ai/wsniady/NAS%20Hyperparameter%20Tuning?nw=nwuserwsniady



## Project Structure

`main.py`
The main script to run the NAS optimizer. It initializes the data loaders, sets up the NAS optimizer, and runs the optimization process.

`NASOptimizer.py`
Contains the NASOptimizer class, which implements the evolutionary algorithm for neural architecture search.

`Chromosome.py`
Defines the Chromosome class, representing a neural network architecture. It includes methods for mutation, crossover, and validation of the architecture.

`Layers.py`
Defines various layer classes (LinearLayer, Conv2dLayer, MaxPool2dLayer, DropoutLayer, ReluLayer) used in the neural network architectures.

`data_loaders.py`
Provides functions to load and preprocess datasets (`MNIST`, `CIFAR-10`, `SVHN`).

`training_utils.py`
Contains utility functions for training and evaluating models, as well as saving the best models.

`get_trained_models.py`
Script to train the saved models and log the results to wandb.

`run_sweep.py`
Script to run a hyperparameter sweep using `wandb`.

`test.py`
Contains test functions to validate the functionality of the Chromosome class and the NAS optimizer.


