# A Project regarding the following Kaggle competition:
# https://www.kaggle.com/c/mercari-price-suggestion-challenge

# Submitted by Yuval Helman and Jakov Zingerman

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import pickle

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



def XGBoosting(subsample, max_depth, min_samples_split, learning_rate, eval_metric, parallel_num=1):
    # specify parameters via map, definition are same as c++ version

    param = {'min_samples_split': min_samples_split, 'max_depth': max_depth, 'learning_rate': learning_rate,
            'eval_metric': eval_metric, 'silent': 1, 'subsample': subsample, 'num_parallel_tree': parallel_num}

    model = XGBRegressor()


    # specify validations set to watch performance
    # watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    # num_round = 2
    # bst = xgb.train(param, dtrain, num_round, watchlist)

    return model


def build_models():
    boost_RF_model = XGBoosting(0.7, # subsample
                                100, # max_depth
                                5, # min_samples_split
                                0.09, # learning_rate
                                'mae', # eval_metric
                                5) # num_parallel_tree
    XGboost_model = XGBoosting(0.7,  # subsample
                                100,  # max_depth
                                5,  # min_samples_split
                                0.09,  # learning_rate
                                'mae',  # eval_metric
                                1)  # num_parallel_tree
