from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from lstm import LSTM
from utils import AverageMeter, accuracy
from torch.utils.data import Subset
import random

CONFIG_DEFAULT = {
    'input_length': 3,
    'input_dim': 1,
    'num_classes': 10,
    'num_hidden': 128,
    'batch_size': 128,
    'learning_rate': 0.001,
    'max_epoch': 50,
    'max_norm': 10,
    'data_size': 100000,
    'portion_train': 0.8,
}


def train(model, data_loader, optimizer, criterion, device, config):
    # TODO set model to train mode
    model.train()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config['max_norm'])

        optimizer.step()
        accu = accuracy(outputs, batch_targets)
        losses.update(loss.item())
        accuracies.update(accu)
        # Add more code here ...
        if step % 10 == 0:
            print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config):
    # TODO set model to evaluation mode
    model.eval()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        accu = accuracy(outputs, batch_targets)
        losses.update(loss.item())
        accuracies.update(accu)

        if step % 10 == 0:
            print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


def split_dataset(dataset, train_ratio=0.8, seed=42):
    """
    Splits a dataset into training and validation subsets.

    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float): The ratio of the dataset to use for training (default: 0.8).
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        train_dataset (Subset): The training subset.
        val_dataset (Subset): The validation subset.
    """
    random.seed(seed)
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def main(config, csv_file):
    if config is None:
        config = CONFIG_DEFAULT
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_length = config['input_length']
    input_dim = config['input_dim']
    num_classes = config['num_classes']
    num_hidden = config['num_hidden']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    max_epoch = config['max_epoch']
    max_norm = config['max_norm']
    data_size = config['data_size']
    portion_train = config['portion_train']
    # Initialize the model that we are going to use
    model = LSTM(input_length, input_dim, num_hidden, num_classes,device)
    model.to(device)
    # Initialize the dataset and data loader
    dataset = PalindromeDataset(input_length, data_size)
    # Split dataset into train and validation sets
    train_dataset, val_dataset = split_dataset(dataset, portion_train)
    # Create data loaders for training and validation
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    results = []

    for epoch in range(max_epoch):
        # Train the model for one epoch
        train_loss, train_acc = train(
            model, train_dloader, optimizer, criterion, device, config)

        scheduler.step()

        # Evaluate the trained model on the validation set
        val_loss, val_acc = evaluate(
            model, val_dloader, criterion, device, config)

        # Save metrics
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

    save_results_to_csv(results, csv_file)

    print('Done training.')


def save_results_to_csv(results, file_path):
    """Save training results to a CSV file."""
    keys = results[0].keys()  # Get column headers from the first result
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=19,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int,
                        default=100, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int,
                        default=100000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8,
                        help='Portion of the total dataset used for training')


    csv_file = 'result/t3.csv'
    config = None
    # Train the model
    main(config, csv_file)
