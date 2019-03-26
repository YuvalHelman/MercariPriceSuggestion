# A Project regarding the following Kaggle competition:
# https://www.kaggle.com/c/mercari-price-suggestion-challenge

# Submitted by Yuval Helman and Jakov Zingerman

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import xgboost as xgb
import pickle
import pandas as pd
from sklearn.metrics import explained_variance_score

def grid_search(classifier, arguments, data, n_fold, default_args=None):
    '''
    Returns the an instance of the classifier, initialized with the best configuration.
    n_fold is the number of cross validation folds to use. You may use GridSearchCV.
    '''
    print("Started grid_search")
    y_train = data['price']
    X_train = data.drop(columns=['price'])
    if default_args:
        model = classifier(**default_args)
    else:
        model = classifier()
    gs = GridSearchCV(model, arguments, cv=n_fold, scoring="accuracy")
    print("After grid_search initialization")
    gs.fit(X_train, y_train.values.ravel())
    print("After grid_seach fit")
    best_arguments = gs.best_params_
    if default_args is not None:
      return classifier(**best_arguments, **default_args)
    return classifier(**best_arguments)


def XGboost_builder(data):
    arguments = {'max_depth': [2, 3], 'learning_rate': [1e-5, 1e-1], 'silent': [True]}

    n_fold = 5
    # xgb grid_search
    clf = grid_search(xgb.XGBClassifier, arguments, data, n_fold)

    return clf


def random_forest_builder(data):
    arguments = {'n_estimators': [100], 'criterion': ['mse', 'mae'], 'max_depth': ['20', '40'], 'min_samples_split': [2,5]}

    n_fold= 5

    clf = grid_search(RandomForestRegressor, arguments, data, n_fold)

    print("Random Forest builder initiated")

    return clf



def XGBoosting(subsample, max_depth, min_samples_split, learning_rate, eval_metric, parallel_num=1, n_trees=10):
    # specify parameters via map, definition are same as c++ version

    param = {'min_samples_split': min_samples_split, 'max_depth': max_depth, 'learning_rate': learning_rate,
            'eval_metric': eval_metric, 'silent': 1, 'subsample': subsample, 'num_parallel_tree': parallel_num
             , 'n_estimators': n_trees}

    model = XGBRegressor(**param)


    # specify validations set to watch performance
    # watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    # num_round = 2
    # bst = xgb.train(param, dtrain, num_round, watchlist)

    return model


def build_models(x_train, y_train):
    # boost_RF_model, param = XGBoosting(0.7, # subsample
    #                             20, # max_depth
    #                             5, # min_samples_split
    #                             0.09, # learning_rate
    #                             'mae', # eval_metric
    #                             5,  # num_parallel_tree
    #                             15 ) # number of trees
    #
    # boost_RF_model.fit(X=x_train, y=y_train)
    # boost_RF_model.save_model('RF_model')

    XGboost_model = XGBoosting(0.7,  # subsample
                                20,  # max_depth
                                5,  # min_samples_split
                                0.09,  # learning_rate
                                'mae',  # eval_metric
                                1, # num_parallel_tree
                                15 )  # number of trees

    XGboost_model.fit(X=x_train, y=y_train)
    XGboost_model.save_model('xgboost_model')


if __name__ == '__main__':
    test_data = pd.read_csv('./x_test.csv')
    test_labels = pd.read_csv('./y_test.csv')
    # train_data = pd.read_csv('./x_train.csv')
    # train_labels = pd.read_csv('./y_train.csv')

    # build_models(train_data, train_labels)

    ''' Load the models from their files '''
    XGboost_model = XGBRegressor()
    XGboost_model.load_model('xgboost_model')
    # boost_RF_model = XGBRegressor()
    # boost_RF_model.load_model('RF_model')
    #

    '''' Initiate score check on the XGBoost model '''
    predictions = XGboost_model.predict(test_data)
    print(test_labels.values())
    labels_arr = test_labels.to_numpy().reshape(-1)
    print(explained_variance_score(predictions, test_labels))