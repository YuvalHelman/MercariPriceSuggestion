import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd
import itertools.product


class Net(nn.Module):

    def __init__(self, first_hidden_input_dim, second_hidden_input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(500, first_hidden_input_dim)
        self.fc2 = nn.Linear(first_hidden_input_dim, second_hidden_input_dim)
        self.fc3 = nn.Linear(second_hidden_input_dim, 256)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = nn.Linear(256, 1, self.fc3(x))
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
        for i, data_vector in enumerate(data):
            optimizer.zero_grad()
            outputs = net(data_vector)
            loss = loss_func(outputs, real_labels)
            loss.backward()
            optimizer.step()

    print("Finished training")
    return net


def test_network(net, test_data, test_labels):
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = net(test_data)
        _, predicted = torch.max(outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    return 100 * correct / total


if __name__ == '__main__':
    first_hidden_input_dim = 256
    second_hidden_input_dim = 256
    net = Net(first_hidden_input_dim, second_hidden_input_dim)
    train_data = pd.read_csv("reduced_train.csv")
    train_labels = []  # TODO: fix
    test_data = []  # TODO: fix
    test_labels = []  # TODO: fix
    loss_funcs = [nn.CrossEntropyLoss(), nn.MSELoss()]  # TODO: try also with other loss functions
    optimizers = [optim.SGD(net.parameters(), lr=0.001,
                            momentum=0.9)]
    # TODO: add other optimizers (USE ADAM also (adagrad), seems good. will write on the word's sheet that we need
    losses_and_optimizers = itertools.product(loss_funcs, optimizers)  # We will try all combinations
    accuracies = []
    for loss_optim in losses_and_optimizers:
        net = train_network(net, loss_optim[0], loss_optim[1], train_data, train_labels, num_epochs=2)
        accuracies.append(test_network(net, test_data, test_labels))
        print("Accuracy for ({0},{1}) is {2}%".format(loss_optim[0].__class__, loss_optim[1].__class__,
                                                      accuracies[len(accuracies) - 1]))

    # we will now compare the results to other ML algorithms
    model = RandomForestRegressor()
    model.fit(train_data, train_labels)

    # Get the mean absolute error on the validation data
    predicted_prices = model.predict(test_data)
    MAE = mean_absolute_error(test_labels, predicted_prices)
    print('Random forest validation MAE = ', MAE)

    # And xgboost
    XGBModel = XGBRegressor()
    XGBModel.fit(train_data, train_labels, verbose=False)

    # Get the mean absolute error on the validation data :
    XGBpredictions = XGBModel.predict(test_data)
    MAE = mean_absolute_error(test_labels, XGBpredictions)
    print('XGBoost validation MAE = ', MAE)
