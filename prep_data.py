# A Project regarding the following Kaggle competition:
# https://www.kaggle.com/c/mercari-price-suggestion-challenge

# Submitted by Yuval Helman and Jakov Zingerman

from mercariPriceData.InferSent.encoder.models import InferSent  # change folders
import pandas as pd
import testing_the_data as tests
import torch
import nltk
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


#################################################
# Global Variables:
puredata = pd.DataFrame()
#################################################



''' series_to_encode: a 'series' type to be transfered to vectors by infersent '''
''' batch_size_to_encode: number of sentences to encode each time (so we don't run out of RAM) '''
''' return: a dataframe of the sentences encodings to 4096 length vectors'''
# https://github.com/facebookresearch/InferSent
def infersent_encoder(series_to_encode, batch_size_to_encode):
    sentences = series_to_encode.tolist()

    nltk.download('punkt')
    V = 2
    MODEL_PATH = './mercariPriceData/InferSent/encoder/infersent%s.pickle' % V  # change folders
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    W2V_PATH = './mercariPriceData/dataset/fastText/cc.en.300.vec'  # change folders
    infersent.set_w2v_path(W2V_PATH)
    try:
        infersent.build_vocab(sentences, tokenize=True)
        print("done build vocab")
    except Exception as e:
        print('build vocab failed')
        print(e)

    # Cant Encode all at once.. (not enough RAM) , so we need to do it in batches
    full_embeddings = [list(np.zeros(4096))]
    end_index, start_index, embeddings = 0, 0, 0
    try:
        # Iterate sentences and encode on batches
        start_index = 0
        end_index = start_index + batch_size_to_encode

        print("number of sentences total: ", len(sentences))
        while end_index < len(sentences):
            part_of_sentences = sentences[start_index:end_index].copy()  # 0 to 1999
            embeddings = infersent.encode(part_of_sentences, tokenize=True)
            # Iteration phase:
            start_index = end_index
            end_index = start_index + batch_size_to_encode
            full_embeddings = np.append(full_embeddings, embeddings, axis=0)
            # full_embeddings = np.concatenate((full_embeddings, embeddings))

        # when end_index is bigger-equal to the length, do a last encoding manually
        end_index = len(sentences)
        if end_index > start_index:
            part_of_sentences = sentences[start_index:end_index].copy()
            embeddings = infersent.encode(part_of_sentences, tokenize=True)
            full_embeddings = np.append(full_embeddings, embeddings, axis=0)

    except Exception as e:
        print('encoding failed on part of list: ', start_index, end_index)
        print(e)

    full_embeddings = full_embeddings[1:]
    full_embeddings = pd.DataFrame.from_records(full_embeddings)
    return full_embeddings

def data_preprocessing(data, start_index, end_index):
    """
    A preprocessing of the data, using InferSent, One-Hot encodings and arranging NaN's etc.
    Returns: a new DataFrame with only numerical data. column length: [ (4096*4) + 2 + 5
    """
    data = data.iloc[start_index:end_index].copy()  # TODO: erase that for doing for all data

    data.reset_index(inplace=True, drop=True)

    # Change anything with Nan \ Not-A-String to an empty string..
    for row_index, val in enumerate(data['item_description']):
        if (isinstance(val, str) == False):
            col_index = data.columns.get_loc("item_description")
            # print(data.iat[row_index, col_index])
            data.iat[row_index, col_index] = 'not known'
            # print("after: ", data.iat[row_index, col_index])
    data["item_description"].fillna("not known", inplace=True)
    for row_index, val in enumerate(data['name']):
        if (isinstance(val, str) == False):
            col_index = data.columns.get_loc("name")
            # print(data.iat[row_index, col_index])
            data.iat[row_index, col_index] = 'not known'
            # print("after: ", data.iat[row_index, col_index])
    data["name"].fillna("not known", inplace=True)
    for row_index, val in enumerate(data['category_name']):
        if (isinstance(val, str) == False):
            col_index = data.columns.get_loc("category_name")
            # print(data.iat[row_index, col_index])
            data.iat[row_index, col_index] = 'not known'
            # print("after: ", data.iat[row_index, col_index])
    data["category_name"].fillna("not known", inplace=True)
    for row_index, val in enumerate(data['brand_name']):
        if (isinstance(val, str) == False):
            col_index = data.columns.get_loc("brand_name")
            # print(data.iat[row_index, col_index])
            data.iat[row_index, col_index] = 'not known'
            # print("after: ", data.iat[row_index, col_index])
    data["brand_name"].fillna("not known", inplace=True)

    # ___________________________________________________________________________________________________
    # Using one-hot encoding on the shipping and item_condition_id columns
    data = pd.concat([data, pd.get_dummies(data['shipping'], prefix='shipping')], axis=1)
    data.drop(columns='shipping', inplace=True)

    data = pd.concat([data, pd.get_dummies(data['item_condition_id'], prefix='item_condition_id')], axis=1)
    data.drop(columns='item_condition_id', inplace=True)
    # ___________________________________________________________________________________________________
    # Using infersent on the item_description column in order to transpose it to vectors (size: 4096)

    data.drop(['train_id'], axis=1, inplace=True)


    row_indices_list = range(start_index,end_index)
    batch_size_to_encode = 5 # TODO: for all data encoding

    new_vectors_df = data.copy() # Start working with new_vectors_df

    start = time.time()
    description_embeddings = infersent_encoder(pd.Series(data["item_description"]), batch_size_to_encode)
    #description_embeddings.index = row_indices_list  # needed for append
    ############################################################
    new_vectors_df.drop(['item_description'], axis=1, inplace=True)
    new_vectors_df = pd.concat([new_vectors_df, description_embeddings], axis=1)
    ############################################################
    print("done encoding item_description")
    end = time.time()
    print("Time for this encoding: ", end - start)

    start = time.time()
    description_embeddings = infersent_encoder(pd.Series(data["name"]), batch_size_to_encode)
    # description_embeddings.index = row_indices_list  # needed for append
    ############################################################
    new_vectors_df.drop(['name'], axis=1, inplace=True)
    new_vectors_df = pd.concat([new_vectors_df, description_embeddings], axis=1)
    ############################################################
    print("done encoding name")
    end = time.time()
    print("Time for this encoding: ", end - start)

    start = time.time()
    description_embeddings = infersent_encoder(pd.Series(data["category_name"]), batch_size_to_encode)
    # description_embeddings.index = row_indices_list  # needed for append
    ############################################################
    new_vectors_df.drop(['category_name'], axis=1, inplace=True)
    new_vectors_df = pd.concat([new_vectors_df, description_embeddings], axis=1)
    ############################################################
    print("done encoding category_name")
    end = time.time()
    print("Time for this encoding: ", end - start)

    start = time.time()
    description_embeddings = infersent_encoder(pd.Series(data["brand_name"]), batch_size_to_encode)
    #description_embeddings.index = row_indices_list  # needed for append
    ############################################################
    new_vectors_df.drop(['brand_name'], axis=1, inplace=True)
    new_vectors_df = pd.concat([new_vectors_df, description_embeddings], axis=1)
    ############################################################
    print("done encoding brand_name")
    end = time.time()
    print("Time for this encoding: ", end - start)


    new_vectors_df.to_csv('./numeric_train_0_100k.csv', index=False, header=False)
    #
    # with open('./numeric_train_appended.csv', 'a') as file:
    #     new_vectors_df.to_csv(file, index=False, header=False)
    # ___________________________________________________________________________________________________

    return data


def build_numerical_data(data):
    start = time.time()


    start_index, end_index = 20 , 40
    data_preprocessing(data, start_index, end_index)

    ##### Save training data into a CSV (only on first batch):
   # data.to_csv('./numeric_train_appended.csv', index=False, header=True)

    end = time.time()
    print("Time for total data preprocessing: ", end - start)


''' 
Splitting the labels and the predictive data and using PCA to reduce dimensionality
'''
def get_reduced_data(data, reduced_dim=500):
    # Get labels outside

    labels = data.loc[:, 'price'].copy()
    data.drop(['price'], axis=1, inplace=True)

    pca = PCA(n_components=reduced_dim)
    Reduced_data = pca.fit_transform(data)
    Reduced_data = pd.DataFrame(Reduced_data)

    X_train, X_test, y_train, y_test = train_test_split(
        Reduced_data, labels, test_size=0.33, random_state=None)

    print("data shape before: ", data.shape)
    print("data shape after: ", Reduced_data.shape)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':

    '''fetching the data and transforming it into numerical data using Infersent '''
    puredata = pd.read_csv('./mercariPriceData/dataset/train.tsv', sep='\t', encoding="utf_8")  # change folders
    build_numerical_data(puredata)

    ######### DEBUG ###############################
    # try_data = pd.read_csv('./numeric_train_appended.csv, header=0)  # change folders
    # print(try_data.shape)
    # ##############################################


    ''' Write the numeric data into a CSV, then use PCA on it and save to another CSV (split to test/train) '''
    # n_data = pd.read_csv('./numeric_train.csv')  # change folders
    #print(n_data.shape)
    # X_train, X_test, y_train, y_test = get_reduced_data(n_data, 500)
    # X_test.to_csv('./x_test.csv', encoding='utf_8', index=False, header=True)
    # y_test.to_csv('./y_test.csv', encoding='utf_8', index=False, header=True)
    # X_train.to_csv('./x_train.csv', encoding='utf_8', index=False, header=True)
    # y_train.to_csv('./y_train.csv', encoding='utf_8', index=False, header=True)

