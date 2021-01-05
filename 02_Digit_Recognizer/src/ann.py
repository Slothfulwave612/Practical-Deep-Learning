"""
__author__: Anmol_Durgapal(@slothfulwave612)

Python module to create artificial neural network.
"""

# required packages and module
import numpy as np
import pandas as pd

# from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils

# define a random-seed
RANDOM_SEED = 42


class NeuralNetwork(nn.Module):
    """
    class for creating a neural network.
    """

    def __init__(
        self, input_features, num_hidden_layers, num_hidden_neurons, num_output
    ):
        """
        Function to initialize class objects.

        Args:
            input_features (int): number of input features.
            num_hidden_layers (int): number of hidden layers.
            num_hidden_neurons (int): number of neurons in hidden layers.
            num_output (int): number of output neurons.
        """
        # to access properties and methods of parent class
        super(NeuralNetwork, self).__init__()

        # init current dimension as input dimension
        current_dim = input_features

        # create an object of ModuleList
        self.layers = nn.ModuleList()

        # iterate and create the architecture
        for i in range(num_hidden_layers + 2):

            # append the layer architecture
            self.layers.append(
                nn.Linear(
                    in_features=current_dim, out_features=num_hidden_neurons
                )
            )

            # change current dimensions
            current_dim = num_hidden_neurons

        # append the output layer architecture
        self.layers.append(
            nn.Linear(
                in_features=current_dim, out_features=num_output
            )
        )

    def forward_pass(self, x):
        """
        Function to make forward pass architecture.

        Args:
            x (torch.tensor): input data.

        Return:
            int: output label.
        """

        # iterate over layers list
        for layer in self.layers[:-1]:

            # forward pass with RELU activation
            x = F.relu(layer(x))

        out = F.softmax(self.layers[-1](x), dim=1)

        return out


if __name__ == "__main__":

    # set random_seed
    torch.manual_seed(RANDOM_SEED)

    # load the datasets
    train_df = pd.read_pickle("data/modelling_data/train.pkl")
    valid_df = pd.read_pickle("data/modelling_data/valid.pkl")
    test_df = pd.read_pickle("data/modelling_data/test.pkl")

    # create object of NeuralNetwork
    ann = NeuralNetwork(
        input_features=784, num_hidden_layers=4,
        num_hidden_neurons=32, num_output=10
    )

    utils.model_training_ann(
        ann, train_df, "label", use_gpu=True
    )
