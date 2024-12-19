from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from pytorch_mlp import MLP
import matplotlib.pyplot as plt



from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None


def train(model, train_data, train_labels, X_val, y_val, epochs=1000, learning_rate=0.001, batch_size=64):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # TRAINING CODE GOES HERE
    X_train = torch.tensor(train_data[:, :256], dtype=torch.float32)
    y_train = torch.tensor(train_labels[:, 0] - train_labels[:, 0].min(), dtype=torch.long)

    # Create DataLoader for training data
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # Convert validation data to tensors, if provided
    if X_val is not None and y_val is not None:
        X_val = torch.tensor(X_val[:, :256], dtype=torch.float32)
        y_val = torch.tensor(y_val[:, 0] - y_val[:, 0].min(), dtype=torch.long)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses = []
    epoch_accuracies = []
    val_accuracies = []
    val_losses = []

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        all_preds = []
        all_labels = []
        model.train()  # Set model to training mode

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        # Validation evaluation
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                _, val_pred = torch.max(val_outputs, 1)
                val_accuracy = accuracy_score(y_val.cpu().numpy(), val_pred.cpu().numpy())
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

            print(
                f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        else:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    return epoch_losses, epoch_accuracies, val_losses, val_accuracies


def generate_data():
    x, y = make_moons(n_samples=1000, shuffle=True, noise=0.2, random_state=42)
    one_hot_encoded = OneHotEncoder(sparse_output=False)
    y_2d = y.reshape(-1, 1)
    y_onehot = one_hot_encoded.fit_transform(y_2d)

    x_train, x_test, y_train_onehot, y_test_onehot = train_test_split(x, y_onehot, test_size=0.2, random_state=42)
    return x_train, y_train_onehot, x_test, y_test_onehot


def main():
    """
    Main function
    """

    train_data, train_labels, validation_data, validation_labels = generate_data()
    input_size = train_data.shape[1]
    hidden_sizes = [128, 64, 32]
    output_size = train_labels.shape[1]

    mlp = MLP(input_size, hidden_sizes, output_size)
    print(train_data[:10,])
    print(train_labels[:10, ])
    train_loss, train_accu, test_loss, test_accu = train(mlp, train_data, train_labels, validation_data, validation_labels)
    # Plot training and validation loss and accuracy
    plot_decision_boundary_p(mlp, train_data, train_labels, validation_data, validation_labels)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Validation Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accu, label='Train Accuracy')
    plt.plot(test_accu, label='Validation Accuracy', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_decision_boundary_p(model, x_train, y_train, X_test, y_test, device='cpu'):
    """
    Plot the decision boundary for a classification model.
    Arguments:
        model: The trained model (MLP in this case).
        x_train: Training data.
        y_train: Training labels.
        X_test: Test data.
        y_test: Test labels.
    """
    # Create a meshgrid that spans the data range
    h = 0.02  # Grid step size
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Convert the grid points to tensor for the model
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)

    # Make predictions on the grid
    with torch.no_grad():
        model.eval()
        pred_grid = model(grid_points)
        _, pred_classes = torch.max(pred_grid, 1)

    # Reshape the predictions back into the meshgrid shape
    pred_classes = pred_classes.cpu().numpy().reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, pred_classes, cmap=plt.cm.coolwarm, alpha=0.6)

    # Plot the training points
    plt.scatter(x_train[:, 0], x_train[:, 1], c=np.argmax(y_train, axis=1), s=50, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
    # Plot the test points with a filled marker ('o') and edgecolors
    plt.scatter(X_test[:, 0], X_test[:, 1], c=np.argmax(y_test, axis=1), s=50, edgecolors='k', marker='o',
                cmap=plt.cm.coolwarm)

    plt.title("Decision Boundary and Data Points")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

if __name__ == '__main__':
    main()

