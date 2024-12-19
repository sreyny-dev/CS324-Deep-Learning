from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from cnn_model import CNN
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm as tqdm

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    return accuracy


def train(model,
          train_loader,
          test_loader,
          max_epoch=MAX_EPOCHS_DEFAULT,
          learning_rate=LEARNING_RATE_DEFAULT,
          optimizer_type=OPTIMIZER_DEFAULT
          ):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE

    if optimizer_type == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    elif optimizer_type == 'RMS':
        optimizer = optim.RMSprop(model.parameters(), lr = learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    criterion = nn.CrossEntropyLoss()

    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    for epoch in range(max_epoch):
        model.train()  # Set the model to training mode
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_train_loss += loss.item() * inputs.size(0)  # accumulate training loss
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate training accuracy and loss for this epoch
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_accuracy = correct_train / total_train

        # Now evaluate the model on the test set
        model.eval()  # Set the model to evaluation mode (no gradients)
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():  # No gradients needed for evaluation
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate loss

                running_test_loss += loss.item() * inputs.size(0)  # accumulate test loss
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        # Calculate test accuracy and loss for this epoch
        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        epoch_test_accuracy = correct_test / total_test

        # Store metrics for plotting
        train_accuracies.append(epoch_train_accuracy)
        train_losses.append(epoch_train_loss)
        test_accuracies.append(epoch_test_accuracy)
        test_losses.append(epoch_test_loss)

        print(f'Epoch{epoch+1}/{max_epoch}, Train Accuracy: {epoch_train_accuracy:.4f}, Train Loss: {epoch_train_loss:.4f}, Test Accuracy: {epoch_test_accuracy:.4f}, Test Loss: {epoch_test_loss:.4f}')

    return train_losses, train_accuracies, test_losses, test_accuracies


def main():
    """
    Main function
    """
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    # parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
    #                     help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
