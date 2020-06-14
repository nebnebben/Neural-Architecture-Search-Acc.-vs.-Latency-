import torch
import numpy as np
import torch.nn as nn
import torch.utils as utils
import math
from Actual import constants
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
import torchvision
from sklearn.datasets import make_classification

# This defines the different problems as a general whole to be used


# Standardises tensors
def standardise(x_tensor):
    means = x_tensor.mean(dim=0)
    stds = x_tensor.std(dim=0)
    x_tensor_standardised = (x_tensor - means) / stds
    return x_tensor_standardised

def test_prob():
    # Set random seeds
    if constants.seeded:
        torch.manual_seed(constants.seed_no)
        np.random.seed(seed=constants.seed_no)
    np.random.seed(seed=3)
    # Some prob
    size = 4000
    a = 5.5
    b1, b2, b3 = 8, 3, 9
    x = np.random.rand(size, 4)
    y = a + x[:, 0] * b1 + (x[:, 1] * b2) * (x[:, 2] * b3) + x[:, 3] ** 2 \
        - (x[:, 0] * x[:, 3] * ((x[:, 1] * b2) - (x[:, 2] * b3)))
    # print(y)

    # Set the training data up
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    # Standardise x data
    x_tensor_standardised = standardise(x_tensor)
    dataset = utils.data.TensorDataset(x_tensor_standardised, y_tensor)

    # Sets up training, val and testing data

    split = [int((size/10)*6), int(size/5), int(size/5)]
    train_dataset, val_dataset, test_dataset = utils.data.dataset.random_split(dataset, split)

    train_loader = utils.data.DataLoader(dataset=train_dataset, batch_size=16)
    val_loader = utils.data.DataLoader(dataset=val_dataset, batch_size=20)
    test_loader = utils.data.DataLoader(dataset=test_dataset, batch_size=20)

    return train_loader, val_loader, test_loader


def mnist_prob():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    batch_size = 64
    mnist_train = DataLoader(torchvision.datasets.MNIST(root="", train=True, download=True, transform=transform),
                             batch_size=batch_size)
    mnist_test = DataLoader(torchvision.datasets.MNIST(root="", train=False, download=True, transform=transform),
                            batch_size=batch_size)
    return mnist_train, mnist_test


def hypercube():
    n_samples = 6000
    x, y = make_classification(n_samples=n_samples, n_features=10, n_redundant=0, n_classes=5, n_informative=10,
                                 n_clusters_per_class=1, class_sep=1.2, random_state=0)
    # Set the training data up
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).long()

    # Standardise x data
    x_tensor_standardised = standardise(x_tensor)
    dataset = utils.data.TensorDataset(x_tensor_standardised, y_tensor)

    # Sets up training, val and testing data

    split = [int(n_samples*0.8), int(n_samples*0.2)]
    train_dataset, test_dataset = utils.data.dataset.random_split(dataset, split)

    train_loader = utils.data.DataLoader(dataset=train_dataset, batch_size=40)
    test_loader = utils.data.DataLoader(dataset=test_dataset, batch_size=20)

    return train_loader, test_loader


def make_moons():
    from sklearn.datasets import make_moons
    n_samples = 3600
    x, y = make_moons(n_samples=n_samples, noise=0.15, random_state=constants.seed_no)

    # Set the training data up
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).long()

    # Standardise x data
    x_tensor_standardised = standardise(x_tensor)
    dataset = utils.data.TensorDataset(x_tensor_standardised, y_tensor)

    # Sets up training, val and testing data

    split = [int(n_samples*0.8), int(n_samples*0.2)]
    train_dataset, test_dataset = utils.data.dataset.random_split(dataset, split)

    train_loader = utils.data.DataLoader(dataset=train_dataset, batch_size=40)
    test_loader = utils.data.DataLoader(dataset=test_dataset, batch_size=20)

    return train_loader, test_loader

def complex_reg():
    from numpy import sin, cos, exp
    n_samples = 20000
    x = np.random.rand(n_samples, 8)

    x1, x2, x3, x4, x5, x6, x7, x8 = np.split(x, 8, 1)

    y = 1.5 + 2 * sin(x1) + 3 * cos(x2) + x3 ** x3 ** x3 + 5 * exp(-1 / x4) + x1 * x2 + 6 * x3 * x6 * x7 * x8 - 1 / (
                0.1 + x5 + x8)

    # Set the training data up
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    # Standardise x data
    x_tensor_standardised = standardise(x_tensor)
    dataset = utils.data.TensorDataset(x_tensor_standardised, y_tensor)

    # Sets up training, val and testing data

    split = [int(n_samples*0.8), int(n_samples*0.2)]
    train_dataset, test_dataset = utils.data.dataset.random_split(dataset, split)

    train_loader = utils.data.DataLoader(dataset=train_dataset, batch_size=40)
    test_loader = utils.data.DataLoader(dataset=test_dataset, batch_size=20)

    return train_loader, test_loader

