from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np
import os
from torch import nn
import argparse


class MLP(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()

        # Define the MLP architecture
        # self.model = nn.Sequential(
        #     nn.Linear(n_inputs, n_hidden[0]),
        #     nn.ReLU(),
        #     nn.Linear(n_hidden[0], n_hidden[1]),
        #     nn.ReLU(),
        #     nn.Linear(n_hidden[1], n_hidden[2]),
        #     nn.ReLU(),
        #     nn.Linear(n_hidden[2], n_classes)
        # )

        self.model = nn.Sequential(
            nn.Linear(n_inputs, n_hidden[0]),
            nn.ReLU(),
            nn.Linear(n_hidden[0], n_classes)
        )

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = self.model(x)
        return out


if __name__ == '__main__':
    n_inputs = 10
    n_hidden = [20, 30]
    n_classes = 5
    model = MLP(n_inputs, n_hidden, n_classes)
    print(model)
