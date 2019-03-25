import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class Net(nn.Module):

    def __init__(self, first_hidden_input_dim, second_hidden_input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # TODO: change dimensions
        self.fc2 = nn.Linear(120, 84)  # TODO: change dimensions
        self.fc3 = nn.Linear(84, 10)  # TODO: change dimensions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = nn.linear(256, 1, self.fc3(x))
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
    net = Net()
    loss_funcs = [nn.CrossEntropyLoss(), nn.MSELoss()]  # TODO: try also with other loss functions
    optimizers = [optim.SGD(net.parameters(), lr=0.001, momentum=0.9)]  # TOOD: add other optimizers (USE ADAM also (adagrad), seems good. will write on the word's sheet that we need
    for loss_func in loss_funcs:
        for optimizer in optimizers:
            net = train_network(net, loss_func, optimizer, train_data, train_labels, num_epochs=2)
            accuracy = test_network(net, test_data, test_labels)

    # we will now compare the results to other ML algorithms
    model = RandomForestRegressor()
    model.fit(train_X, train_y)

    # Get the mean absolute error on the validation data
    predicted_prices = model.predict(val_X)
    MAE = mean_absolute_error(val_y, predicted_prices)
    print('Random forest validation MAE = ', MAE)

    # And xgboost
    XGBModel = XGBRegressor()
    XGBModel.fit(train_X, train_y, verbose=False)

    # Get the mean absolute error on the validation data :
    XGBpredictions = XGBModel.predict(val_X)
    MAE = mean_absolute_error(val_y, XGBpredictions)
    print('XGBoost validation MAE = ', MAE)

