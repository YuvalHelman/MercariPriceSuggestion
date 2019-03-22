# A Project regarding the following Kaggle competition:
# https://www.kaggle.com/c/mercari-price-suggestion-challenge

# Submitted by Yuval Helman and Jakov Zingerman

import pandas as pd
from sklearn.decomposition import PCA
import torch
import nltk
import numpy as np


''' 
Splitting the labels outside and using PCA to reduce dimensionality
'''
def get_reduced_data(data, reduced_dim=20):

    # Get labels outside
    labels = data['price'].copy()
    data.drop(['price'], axis=1, inplace=True)

    pca = PCA(n_components=reduced_dim)
    Reduced_data = pca.fit_transform(data)
    Reduced_data = pd.DataFrame(Reduced_data)
    print("data shape: ", data.shape)
    print("data shape: ", Reduced_data.shape)
    print(Reduced_data.head())

    return Reduced_data , labels


