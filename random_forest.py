from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

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
