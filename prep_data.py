# A Project regarding the following Kaggle competition:
# https://www.kaggle.com/c/mercari-price-suggestion-challenge

# Submitted by Yuval Helman and Jakov Zingerman

from mercariPriceData.InferSent.encoder.models import InferSent  # change folders
import pandas as pd
from pandas import DataFrame
import torch
import nltk
import numpy as np
import time
from sklearn.decomposition import PCA


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
        print('build vocab failed. Error occurred during InferSent encoding')
        print(e)

    # Can't Encode all at once.. (not enough RAM) , so we need to do it in batches
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

        # when end_index is bigger-equal to the length, do a last encoding manually
        end_index = len(sentences)
        if end_index > start_index:
            part_of_sentences = sentences[start_index:end_index].copy()
            embeddings = infersent.encode(part_of_sentences, tokenize=True)
            full_embeddings = np.append(full_embeddings, embeddings, axis=0)
    except Exception as e:
        print('encoding failed on part of list: ', start_index, end_index)
        print(e)

    full_embeddings = full_embeddings[1:] # Erase the zero vector

    return full_embeddings

''' Combines the text columns of a row into a single string '''
def string_of_all_text_columns(row):
    Combined_columns = ''

    if (not (row['name'] == 'not known')):
        new_string = Combined_columns + " " + row['name']

    if (not (row['category_name'] == 'not known')):
        new_string = Combined_columns + " " + row['category_name']

    if (not (row['brand_name'] == 'not known')):
        new_string = Combined_columns + " " + row['brand_name']

    if (not (row['item_description'] == 'not known')) and (not (row['item_description'] == 'No description yet')):
        new_string = Combined_columns + " " + row['item_description']

    return Combined_columns


'''
Creates a new column for the dataframe, with a concatenated string of all text columns 
from each row in it.
'''
def combine_text_columns_into_one_later(data):
    ret_data = data.copy()
    ret_data.apply(lambda row: string_of_all_text_columns(row), axis=1)

    ret_data['text_column'] = ret_data.apply(lambda row: string_of_all_text_columns(row), axis=1)

    return ret_data


'''
Iterates on data[start_index:end_index] and creates a processed data.
Combines InferSent, one-hot encodings and everything and then writes this data into a pickle file.
'''
def data_preprocessing(data, start_index, end_index):
    """
    A preprocessing of the data, using InferSent, One-Hot encodings and arranging NaN's etc.
    Returns: a new DataFrame with only numerical data. column length: (4096*4) + 2 + 5
    """
    __start = time.time()

    data = data.iloc[start_index:end_index].copy()

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

    # Combine all of the text columns into one. this is mainly to have the encoding run faster
    data = combine_text_columns_into_one_later(data)
    data.drop(['item_description'], axis=1, inplace=True)
    data.drop(['name'], axis=1, inplace=True)
    data.drop(['category_name'], axis=1, inplace=True)
    data.drop(['brand_name'], axis=1, inplace=True)
    # Drop the train_id column
    data.drop(['train_id'], axis=1, inplace=True)


    row_indices_list = range(start_index,end_index)
    batch_size_to_encode = 5000 # Change for different size of batch to encode. (According to RAM constraints)

    # encode the text column using InferSent. returns a numpy array of shape (rows, 4096)
    description_embeddings = infersent_encoder(pd.Series(data["text_column"]), batch_size_to_encode)
    description_df = DataFrame.from_records(description_embeddings)

    data.drop(['text_column'], axis=1, inplace=True)
    data = pd.concat([data, description_df], axis=1)

    # Write it to a pickle file
    data.to_pickle('./output_pickle_{}_{}.pkl'.format(start_index, end_index), compression='bz2')

    __end = time.time()
    print("Time for total data preprocessing: ", __end - __start)

    return data



''' 
Splitting the labels and the predictive data and using PCA to reduce dimensionality
'''
def get_reduced_data(data, reduced_dim=200):
    # Get labels outside

    labels = data.loc[:, 'price'].copy()
    data.drop(['price'], axis=1, inplace=True)

    pca = PCA(n_components=reduced_dim)
    Reduced_data = pca.fit_transform(data)
    del data
    Reduced_data = pd.DataFrame(Reduced_data)

    return Reduced_data, labels



if __name__ == '__main__':

    '''fetching the data and transforming it into numerical data using Infersent '''
    #puredata = pd.read_csv('./mercariPriceData/dataset/train.tsv', sep='\t', encoding="utf_8")  # folders
    ''' preprocessing only a limited number of the data due to RAM constraints'''
    # start_index, end_index = 200000, 250000  # change this for more data generation
    # data_preprocessing(puredata, start_index, end_index)


    ''' Combining different processed pickle files into combined ones.'''
    #  Combining pickle files into ones.
    # df2 = pd.read_pickle('./output_pickle_25_40.pkl', compression='bz2')
    # df = pd.read_pickle('./output_pickle_200000_250000.pkl', compression='bz2')
    # comb_df = pd.concat([df, df2], ignore_index=True)
    # comb_df.to_pickle('./output_pickle_25_40.pkl', compression='bz2')

    ''' Write the numeric data into a CSV, then use PCA on it and save to another CSV (split to test/train) '''
    # df = pd.read_pickle('./output_pickle_0_15.pkl', compression='bz2')
    # x_train, y_train = get_reduced_data(df, 100)
    #
    # x_train.to_pickle('x_train.pkl', compression='bz2')
    # y_train.to_pickle('y_train.pkl', compression='bz2')

    ''' Concatenate different files of processed data '''
    # df = pd.read_pickle('./output_pickle_15_25.p
    # x_test.to_pickle('x_test.pkl', compression='bz2')
    # y_test.to_pickle('y_test.pkl', compression='bz2')
