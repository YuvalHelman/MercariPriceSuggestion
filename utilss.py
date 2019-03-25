# A Project regarding the following Kaggle competition:
# https://www.kaggle.com/c/mercari-price-suggestion-challenge

# Submitted by Yuval Helman and Jakov Zingerman

import pickle
import xgboost as xgb

''' 
model: the trained model (After fit)
filename: a string containing of the file name (relative to the top directory, or full path)
'''
def save_model_on_file(model, filename, is_XGB=False):
    if is_XGB == True:
        model.save_model(filename)
    else:
        pickle.dump(model, open(filename, 'wb'))

''' 
filename: a string containing of the file name (relative to the top directory, or full path)
'''
def load_model_from_file(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model