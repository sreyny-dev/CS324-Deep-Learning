from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(Model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_inputs, n_hidden[0]),
            nn.ReLU(),
            nn.Linear(n_hidden[0], n_hidden[1]),
            nn.ReLU(),
            # nn.Linear(n_hidden[1], n_hidden[2]),
            # nn.ReLU(),
            nn.Linear(n_hidden[1], n_classes)
        )

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = x.view(-1, 3 * 32 * 32)
        out = self.model(out)
        return out


def train(model, train_loader, test_loader, learning_rate, num_epochs):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # for output
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            pred = model(inputs)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(pred.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_acc = correct_train / total_train
        train_accuracies.append(train_acc)

        # Evaluating on test set
        correct_test = 0
        total_test = 0
        test_running_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images)
                test_running_loss += loss_fn(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()


        test_loss = test_running_loss / len(test_loader)
        test_losses.append(test_loss)
        test_acc = correct_test / total_test
        test_accuracies.append(test_acc)

        # print process
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Val Loss: {test_loss:.4f}, Val Accuracy: {test_acc:.4f}")

    return train_losses, train_accuracies, test_losses, test_accuracies

