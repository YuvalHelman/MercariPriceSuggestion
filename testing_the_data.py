# A Project regarding the following Kaggle competition:
# https://www.kaggle.com/c/mercari-price-suggestion-challenge

# Submitted by Yuval Helman and Jakov Zingerman

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


''' 
Splitting the labels and the predictive data and using PCA to reduce dimensionality
'''
def get_reduced_data(data, reduced_dim=500):
    # Get labels outside

    labels = data.loc[:, 'price'].copy()
    print("done copying")
    data.drop(['price'], axis=1, inplace=True)

    pca = PCA(n_components=reduced_dim)
    Reduced_data = pca.fit_transform(data)
    Reduced_data = pd.DataFrame(Reduced_data)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.33, random_state=None)

    print("data shape: ", data.shape)
    print("data shape: ", Reduced_data.shape)
    print(Reduced_data.head())

    return X_train, X_test, y_train, y_test
