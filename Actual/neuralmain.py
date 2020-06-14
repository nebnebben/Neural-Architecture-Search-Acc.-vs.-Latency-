import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Actual import constants

class GenNet(nn.Module):

    def __init__(self,  layers):
        super(GenNet, self).__init__()
        self.layers = layers

    def forward(self, x):
        if constants.current_problem == "mnist":
            x = x.view(-1, 784)

        for el in self.layers:
            x = el(x)
        return x


# Creates the model from specs got from genetic alg
# Then returns it
# Input:
# layers = list of integers corresponding to activations
# nodes = number of nodes per layer

def createModel(layers):
    if constants.seeded:
        torch.manual_seed(constants.seed_no)
    act_functions = constants.activation_list
    module_list = []
    for i, layer in enumerate(layers):
        # Add node to module list
        # Adds in and out nodes
        nnlayer = nn.Linear(layer[1][0], layer[1][1])
        module_list.append(nnlayer)

        if i < len(layers) - 1:
            batch_norm = nn.BatchNorm1d(layer[1][1])
            module_list.append(batch_norm)
        # Adds function
        module_list.append(act_functions[layer[0]])

        if i < len(layers) - 1:
            drop_out = nn.Dropout(p=0.5)
            module_list.append(drop_out)

    if constants.problem_type == "classification":
        module_list[-1] = nn.Softmax(dim=1)
    module_list = nn.ModuleList(module_list)
    model = GenNet(module_list)
    return model


# layers = [nn.Linear(4, 4), nn.Linear(4, 4)]
# activations = [F.leaky_relu, F.leaky_relu]
# model = neural.GenNet(layers, activations).to(device)



def make_train_step(model):
    if constants.problem_type == "regression":
        loss_fn = nn.MSELoss(reduction='mean')
        optimiser = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    elif constants.problem_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
        optimiser = optim.AdamW(model.parameters())

    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        # loss = loss = loss_fn(y, yhat) ??? if test?
        loss = loss_fn(yhat, y)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimiser.step()
        optimiser.zero_grad()
        # Returns the loss
        return loss.item()

    return train_step

