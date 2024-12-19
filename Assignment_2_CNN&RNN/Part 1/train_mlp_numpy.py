import argparse
import numpy as np
from mlp_numpy import MLPN
from modules import CrossEntropy, Linear
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '128'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1000
EVAL_FREQ_DEFAULT = 1


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.

    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding

    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # TODO: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets

    predicted_result = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    accurate_count = np.sum(predicted_result == true_classes)
    accuracies = accurate_count / predictions.shape[0]
    return accuracies


def train(model,x_train, x_test, y_train, y_test, dnn_hidden_units, learning_rate, max_steps, eval_freq, batch_size):
    """
    Performs training and evaluation of MLP model.

    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        batch_size: Batch size for training (1 for SGD, >1 for batch GD)

    Returns:
        loss_rate: List of loss values on the test set over time
        training_loss: List of loss values on the training set over time
        training_accuracy: List of training accuracy over time
        testing_accuracy: List of testing accuracy over time
    """
    # Load your data
    # Initialize your MLP model and loss function

    loss_fn = CrossEntropy()

    training_accuracy = []
    testing_accuracy = []
    training_loss = []
    testing_loss = []

    for step in range(max_steps):
        if batch_size == 1:
            for i in range(x_train.shape[0]):
                predictions_train = model.forward(x_train[i:i + 1])
                loss = loss_fn.forward(predictions_train, y_train[i:i + 1])
                grads = loss_fn.backward(predictions_train, y_train[i:i + 1])
                model.backward(grads)

                # Update weights and biases
                for l in model.layers:
                    if isinstance(l, Linear):
                        l.params['weight'] -= learning_rate * l.grads['weight']
                        l.params['bias'] -= learning_rate * l.grads['bias']

        else:
            predictions_train = model.forward(x_train)
            loss = loss_fn.forward(predictions_train, y_train)
            grads = loss_fn.backward(predictions_train, y_train)
            model.backward(grads)

            for l in model.layers:
                if isinstance(l, Linear):
                    l.params['weight'] -= learning_rate * l.grads['weight']
                    l.params['bias'] -= learning_rate * l.grads['bias']

        # Evaluate accuracy and loss periodically
        if step % eval_freq == 0 or step == max_steps - 1:

            # Calculate training loss and accuracy
            train_predictions = model.forward(x_train)
            train_loss = loss_fn.forward(train_predictions, y_train)
            train_accuracy = accuracy(train_predictions, y_train)

            # Calculate test loss and accuracy
            test_predictions = model.forward(x_test)
            test_loss = loss_fn.forward(test_predictions, y_test)
            test_accuracy = accuracy(test_predictions, y_test)

            testing_loss.append(test_loss)
            training_loss.append(train_loss)
            testing_accuracy.append(test_accuracy)
            training_accuracy.append(train_accuracy)

            print(
                f"Step: {step+1}, Train Loss: {train_loss:.2f}, Train Accuracy: {train_accuracy:.2f}, Test Loss: {test_loss:.2f}, Test Accuracy: {test_accuracy:.2f}")

    print("Training complete!")
    return training_loss, training_accuracy,testing_loss, testing_accuracy


def generate_data():
    x, y =make_circles (n_samples=1000, shuffle=True, noise=0.2, random_state=42)
    one_hot_encoded = OneHotEncoder(sparse_output=False)
    y_2d = y.reshape(-1, 1)
    y_onehot = one_hot_encoded.fit_transform(y_2d)

    x_train, x_test, y_train_onehot, y_test_onehot = train_test_split(x, y_onehot, test_size=0.2, random_state=42)
    return x_train, x_test, y_train_onehot, y_test_onehot


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def plot_decision_boundary_n(model, x, y, resolution=100):
    """
    Plots the decision boundary of the classifier.

    Args:
        model: The trained model (MLPN instance).
        x: Input data (n_samples, n_features).
        y: Ground truth labels (n_samples,).
        resolution: Grid resolution for contour plot.
    """
    # Create a meshgrid of points
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))

    # Flatten the grid to pass it to the model
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Get predictions for each point in the grid
    predictions = model.forward(grid_points)
    predictions = np.argmax(predictions, axis=1)  # Get predicted class

    # Reshape the predictions to match the grid shape
    predictions = predictions.reshape(xx.shape)

    # Plot the contour map (decision boundary)
    plt.contourf(xx, yy, predictions, alpha=0.6, cmap=plt.cm.coolwarm)

    # Plot the training data
    plt.scatter(x[:, 0], x[:, 1], c=np.argmax(y, axis=1), edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title("Decision Boundary and Training Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()
    plt.show()
def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for training (1 for SGD, >1 for batch GD)')
    FLAGS = parser.parse_known_args()[0]
    x_train, x_test, y_train, y_test = generate_data()
    model = MLPN(x_train.shape[1], list(map(int, FLAGS.dnn_hidden_units.split(','))), y_train.shape[1])
    training_loss, training_accuracy,testing_loss, testing_accuracy = train(model,FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq, FLAGS.batch_size)


if __name__ == '__main__':
    main()
