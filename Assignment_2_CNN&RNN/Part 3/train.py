from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from utils import AverageMeter, accuracy

# Default configuration parameters
DEFAULT_CONFIG = {
    'input_length': 4,
    'input_dim': 1,
    'num_classes': 10,
    'num_hidden_units': 128,
    'batch_size': 128,
    'learning_rate': 0.001,
    'max_epochs': 100,
    'max_grad_norm': 10,
    'dataset_size': 1000000,
    'train_fraction': 0.8,
    'use_lr_scheduler': False,
}


# Training function
def train(model, train_loader, optimizer, loss_fn, device, config):
    model.train()
    loss_meter = AverageMeter("Loss")
    accuracy_meter = AverageMeter("Accuracy")

    for batch_inputs, batch_targets in train_loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        optimizer.zero_grad()
        predictions = model(batch_inputs)
        loss = loss_fn(predictions, batch_targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
        optimizer.step()

        # Calculate accuracy
        acc = accuracy(predictions, batch_targets)
        loss_meter.update(loss.item())
        accuracy_meter.update(acc)

    return loss_meter.avg, accuracy_meter.avg


# Evaluation function
@torch.no_grad()
def evaluate(model, val_loader, loss_fn, device, config):
    model.eval()
    loss_meter = AverageMeter("Loss")
    accuracy_meter = AverageMeter("Accuracy")

    for batch_inputs, batch_targets in val_loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        predictions = model(batch_inputs)
        loss = loss_fn(predictions, batch_targets)
        acc = accuracy(predictions, batch_targets)

        loss_meter.update(loss.item())
        accuracy_meter.update(acc)

    return loss_meter.avg, accuracy_meter.avg


# Main function to set up the model and training process
def main(config):
    if config is None:
        config = DEFAULT_CONFIG

    # Extract configuration parameters
    seq_length = config['input_length']
    input_dim = config['input_dim']
    num_classes = config['num_classes']
    num_hidden_units = config['num_hidden_units']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    max_epochs = config['max_epochs']
    max_grad_norm = config['max_grad_norm']
    dataset_size = config['dataset_size']
    train_fraction = config['train_fraction']
    use_lr_scheduler = config['use_lr_scheduler']

    # Set device to GPU if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {torch.cuda.get_device_name(device)}')

    # Initialize dataset and data loaders
    dataset = PalindromeDataset(seq_length, dataset_size)
    train_dataset, val_dataset = random_split(dataset, [train_fraction, 1 - train_fraction])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = VanillaRNN(seq_length, input_dim, num_hidden_units, num_classes, device)
    model.to(device)

    # Set up loss function and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    # Lists to store training and validation metrics
    train_accuracy_list = []
    val_accuracy_list = []
    train_loss_list = []
    val_loss_list = []

    # Training loop
    for epoch in tqdm(range(max_epochs)):
        # Train for one epoch
        train_loss, train_accuracy = train(model, train_loader, optimizer, loss_fn, device, config)

        # Step the learning rate scheduler if applicable
        if use_lr_scheduler:
            lr_scheduler.step()

        # Evaluate on the validation set
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device, config)

        # Store metrics
        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    # Save results to CSV
    output_filename = 'results/training_results.csv'
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Train Accuracy', 'Val Accuracy', 'Train Loss', 'Val Loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, train_acc, val_acc, train_loss, val_loss in zip(
                range(max_epochs), train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list):
            writer.writerow({'Epoch': epoch, 'Train Accuracy': train_acc, 'Val Accuracy': val_acc,
                             'Train Loss': train_loss, 'Val Loss': val_loss})

    print(f"Results saved to {output_filename}")

    # Final results after training
    print(f'After {max_epochs} epochs:')
    print(
        f'Train Accuracy: {train_accuracy_list[-1] * 100:.4f}%, Validation Accuracy: {val_accuracy_list[-1] * 100:.4f}%')
    print(f'Train Loss: {train_loss_list[-1]:.6f}, Validation Loss: {val_loss_list[-1]:.6f}')

    return train_accuracy_list, val_accuracy_list, train_loss_list, val_loss_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--input_length', type=int, default=19, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--num_hidden_units', type=int, default=128, help='Number of hidden units')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--dataset_size', type=int, default=1000000, help='Total dataset size')
    parser.add_argument('--train_fraction', type=float, default=0.8, help='Fraction of dataset used for training')

    # Parse arguments and run the main function
    config = parser.parse_args()
    main(None)
