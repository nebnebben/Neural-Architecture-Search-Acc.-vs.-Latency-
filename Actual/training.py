# tests model
import torch
import numpy as np
import torch.nn as nn
import torch.utils as utils
from Actual import problems
from Actual import neuralmain as neural
from Actual import constants
import sys

# Runs training and returns loss for regression problems
def run_training(model, train_loader, test_loader):
    # Sets loss function and train step
    train_step = neural.make_train_step(model)
    loss_fn = nn.MSELoss(reduction='mean')

    # Gets the number of epochs
    n_epochs = constants.n_epochs
    device = constants.device
    losses = []

    # For each epoch trains model
    for epoch in range(n_epochs):
        # Trains model
        for x_batch, y_batch in train_loader:
            # Send dataset to GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # Performs one train step and returns corresponding loss
            y_batch = y_batch.view(y_batch.shape[0], 1)
            loss = train_step(x_batch, y_batch)

            losses.append(loss)

    # Tests model
    test_losses = []
    for x_val, y_val in test_loader:
        y_val = y_val.view(y_val.shape[0], 1)
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        model.eval()

        yhat = model(x_val)
        test_loss = loss_fn(y_val, yhat)
        test_losses.append(test_loss.item())

    return np.mean(test_losses)


# Runs training and returns loss for classification problems
def class_train(model, train_loader,  test_loader):
    train_step = neural.make_train_step(model)

    losses = []
    n_epochs = constants.n_epochs
    device = constants.device

    for epoch in range(n_epochs):
        losses = []
        for x_batch, y_batch in train_loader:
            # Send dataset to GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # Performs one train step and returns corresponding loss
            # y_batch = y_batch.view(y_batch.shape[0], 1)
            loss = train_step(x_batch, y_batch)

            # print(loss)
            losses.append(loss)
        # print("Training losses are: ", np.mean(losses))

    test_losses = []
    # Gets test accuracy
    for x_val, y_val in test_loader:
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        y_val = y_val.view(y_val.shape[0], 1)

        model.eval()

        yhat = model(x_val)

        _, out = torch.max(yhat, 1)
        total = len(yhat)
        y_val = y_val.flatten()
        correct = (out == y_val).sum()
        accuracy = int(correct) / total

        test_loss = 1 - accuracy
        test_losses.append(test_loss)

    return np.mean(test_losses)


# Determines which model/dataset is being tested
def modeltest(model):
    # Get the device
    device = constants.device
    # Set the training data up
    model = model.to(device)
    if constants.current_problem == "test":
        train_loader, val_loader, test_loader = problems.test_prob()
        loss = run_training(model, train_loader, test_loader)

    elif constants.current_problem == "mnist":
        train_loader, test_loader = problems.mnist_prob()
        loss = class_train(model, train_loader, test_loader)

    elif constants.current_problem == "hypercube":
        train_loader, test_loader = problems.hypercube()
        loss = class_train(model, train_loader, test_loader)

    elif constants.current_problem == "moons":
        train_loader, test_loader = problems.make_moons()
        loss = class_train(model, train_loader, test_loader)

    elif constants.current_problem == "complex_reg":
        train_loader, test_loader = problems.complex_reg()
        loss = run_training(model, train_loader, test_loader)


    return loss
