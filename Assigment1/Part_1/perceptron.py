import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=100, learning_rate=0.01):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias).
        """
        self.n_inputs = n_inputs
        self.max_epochs = max_epochs  # Fill in: Initialize maximum number of epochs
        self.learning_rate = learning_rate  # Fill in: Initialize learning rate
        self.weights = np.zeros(n_inputs + 1)  # Fill in: Initialize weights with zeros

    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            int: Predicted label (1 or -1) or Predicted lables.
        """
        # Insert bias at the start
        input_with_bias = np.insert(input_vec, 0, 1)  # Add bias term
        return np.sign(np.dot(self.weights, input_with_bias))

    def train(self, training_inputs, training_labels, testing_inputs, testing_labels):

        training_accuracy = []
        testing_accuracy = []
        training_loss = []
        testing_loss = []
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
        """

        # we need max_epochs to train our model
        for _ in range(self.max_epochs):
            """
                What we should do in one epoch ?
                you are required to write code for
                1.do forward pass
                2.calculate the error
                3.compute parameters' gradient
                4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
                please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
            for i in range(len(training_inputs)):
                print(training_inputs.shape)
                prediction = self.forward(training_inputs[i])
                if prediction * training_labels[i] <= 0:
                    self.weights += self.learning_rate * training_labels[i] * np.insert(training_inputs[i], 0, 1)

            train_accuracy = self.calculate_accuracy(training_inputs, training_labels) * 100
            test_accuracy = self.calculate_accuracy(testing_inputs, testing_labels) * 100
            train_loss = (1 - train_accuracy) * 100
            test_loss = (1 - test_accuracy) * 100

            training_accuracy.append(train_accuracy)
            testing_accuracy.append(test_accuracy)
            training_loss.append(train_loss)
            testing_loss.append(test_loss)

        return training_accuracy, testing_accuracy, training_loss, testing_loss

    def calculate_accuracy(self, input, label):
        predictions = np.array([self.forward(x) for x in input])
        return np.mean(predictions == label)


def generate_data(m1, m2, v1,v2):

    x_1 = np.random.normal(m1, v1, (100, 2))
    x_2 = np.random.normal(m2, v2, (100, 2))

    train = np.concatenate((x_1[:80], x_2[:80]), axis=0)
    label_1 = np.ones(80)
    label_2 = -np.ones(80)
    label = np.concatenate((label_1, label_2), axis=0)

    indices = np.random.permutation(len(label))
    train = train[indices]
    label = label[indices]

    test = np.concatenate((x_1[80:], x_2[80:]), axis=0)
    test_label = np.append(np.ones(20), -np.ones(20))

    return train, label, test, test_label


def moving_smooth(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


if __name__ == '__main__':

    mean_1, mean_2 = 2, 4
    var_1, var_2 = 1, 1
    x_train, y_train, x_test, y_test = generate_data(mean_1, mean_2, var_1, var_2)

    perceptron = Perceptron(n_inputs=2)

    training_accu, testing_accu, training_loss, testing_loss = perceptron.train(x_train, y_train, x_test, y_test)
    print(training_accu)
    print(testing_accu)

    window_size = 5
    smoothed_training_accu = moving_smooth(training_accu, window_size)
    smoothed_testing_accu = moving_smooth(testing_accu, window_size)

    epochs = np.arange(len(training_accu))

    # Create smooth x values
    x_new = np.linspace(epochs.min(), epochs.max(), 300)  # 300 points for smoothness

    # Interpolating the accuracy data
    spl_training = make_interp_spline(epochs, training_accu, k=3)  # Cubic spline
    spl_testing = make_interp_spline(epochs, testing_accu, k=3)  # Cubic spline
    spl_training_loss = make_interp_spline(epochs, training_loss, k=3)  # Cubic spline
    spl_testing_loss = make_interp_spline(epochs, testing_loss, k=3)  # Cubic spline

    training_accu_smooth = spl_training(x_new)
    testing_accu_smooth = spl_testing(x_new)
    training_loss_smooth = spl_training_loss(x_new)
    testing_loss_smooth = spl_testing_loss(x_new)
