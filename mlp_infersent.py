import math
import re
import string
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn


def preprocessor(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    text = regex.sub(" ", text)  # remove punctuation
    return text


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


# A function to calculate Root Mean Squared Logarithmic Error (RMSLE) as requested in the challenge
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.view(-1, x.size(0))
        x = self.layers(x)
        return x


if __name__ == '__main__':

    x_train = pd.read_pickle('../mercariData/x_train.pkl', compression='bz2')
    x_train = torch.FloatTensor(x_train.values)
    x_test = pd.read_pickle('../mercariData/x_test.pkl', compression='bz2')
    x_test = torch.FloatTensor(x_test.values)
    y_train = pd.read_pickle('../mercariData/y_train.pkl', compression='bz2')
    y_train = torch.FloatTensor(y_train.values)
    y_test = pd.read_pickle('../mercariData/y_test.pkl', compression='bz2')
    y_test = torch.FloatTensor(y_test.values)
    print("Loaded data")
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.MSELoss()
    print("Initialized model, optimizer and loss function")
    mean_train_losses = []
    mean_valid_losses = []
    valid_acc_list = []
    epochs = 15

    # lets train!
    for epoch in range(epochs):
        print("Training epoch: {0}".format(epoch + 1))
        model.train()

        train_losses = []
        valid_losses = []
        for i, data in enumerate(x_train):
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, y_train[i])
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

    print(train_losses)
    # evaluating the results
    model.eval()
    test_preds = torch.LongTensor()

    for i, data in enumerate(x_test):
        outputs = model(data)

        pred = outputs.max(1, keepdim=True)[1]
        test_preds = torch.cat((test_preds, pred), dim=0)

    test_score = mean_squared_error(y_test, test_preds)
    rmsle_score = rmsle(y_test, test_preds)
    print("The test MSE score is: %d" % test_score)
    print("The test RMSLE score is: %d" % rmsle_score)
