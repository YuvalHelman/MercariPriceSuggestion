import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self, first_hidden_input_dim, second_hidden_input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(500, first_hidden_input_dim)
        self.fc2 = nn.Linear(first_hidden_input_dim, second_hidden_input_dim)
        self.fc3 = nn.Linear(second_hidden_input_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_network(net, loss_func, optimizer, data, real_labels, num_epochs=1):
    """
    gets an initialized network object with an initialized optimizer and preforms an sgd update on it.
    :param net:
    :param loss:
    :param data:
    :param optimizer:
    :return:
    """
    for epoch in range(num_epochs):
        outputs = net(data)
        loss = loss_func(outputs, real_labels)
        print(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Finished training")
    return net


def test_network(net, test_data, test_labels: torch.Tensor):
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = net(test_data)

    print("Test outputs are:")
    print(outputs)

    differences = []
    for i, prediction in outputs:
        differences.append(torch.abs(prediction - test_labels[i]))

    print(differences)
    return differences


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_linear = torch.nn.Linear(500, 20)
        self.second_linear = torch.nn.Linear(20, 20)
        self.third_linear = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = self.first_linear(x)
        x = self.second_linear(x)
        y_pred = self.third_linear(x)
        return y_pred


# print(len(list(model.parameters())))
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    test_data = torch.FloatTensor(pd.read_csv('./x_test.csv').values)
    test_labels = torch.FloatTensor(pd.read_csv('./y_test.csv').values)
    train_data = torch.FloatTensor(pd.read_csv('./x_train.csv').values)
    train_labels = torch.FloatTensor(pd.read_csv('./y_train.csv').values)
    print("Successfully read csv files.")

    model = Model()

    loss_funcs = [nn.MSELoss(), nn.L1Loss()]  # we will iterate them to find the best
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.000001), torch.optim.Adam(model.parameters(), lr=0.000001)]
    loss_optim = list(itertools.product(loss_funcs, optimizers))
    predictions = []
    ### TRAINING
    for loss_func, optimizer in loss_optim:
        losses = []
        for epoch in range(10000):
            y_pred = model(train_data)
            loss = loss_func(y_pred, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(loss_func.__class__, optimizer.__class__)
        print("Achieved loss of:", losses[len(losses) - 1])
        y_test_pred = model(test_data)
        predictions.append(y_test_pred.detach().numpy())
        print("The predictions are:", y_test_pred)

    plt.scatter(np.arange(1, 10 * len(test_labels), 10), test_labels, color='b', s=1)
    i_to_c = ['g', 'r', 'y', 'black']
    for i, pred in enumerate(predictions):
        plt.xlabel("Given values")
        plt.ylabel("Predictions vs Real price values")
        print(len(test_data))
        print(len(pred))
        print(len(test_labels))
        plt.scatter(np.arange(1, 10 * len(pred), 10), pred, color=i_to_c[i], s=1,
                    label="{loss_func} with {optimizer}".format(loss_func=loss_optim[i][0],
                                                                 optimizer=loss_optim[i][1]))
    plt.legend(loc='upper left', prop={'size':6})
    plt.show()
