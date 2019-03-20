# A Project regarding the following Kaggle competition:
# https://www.kaggle.com/c/mercari-price-suggestion-challenge

# Submitted by Yuval Helman and Jakov Zingerman

import fasttext
from ..mercariPriceData.InferSent.encoder.models import InferSent
import pandas as pd
import torch
import nltk
import numpy as np

puredata = pd.read_csv('../mercariPriceData/dataset/train.tsv', sep='\t', encoding="utf_8")


def show_data_structure():
    f = data
    print('#################################################')
    print('LOOKING ON THE DATA STRUCTURE:')
    print('#################################################')
    print('data size: ', len(f))
    print(f['name'].unique())

    print("LOOKING ON NUMBER OF UNIQUE VALUES IN EVERY FEATURE:")
    print('item_condition_id: ', len(f['item_condition_id'].unique()))
    print('category_name: ', len(f['category_name'].unique()))
    print('brand_name: ', len(f['brand_name'].unique()))
    print('price: ', len(f['price'].unique()))
    print('shipping: ', len(f['shipping'].unique()))
    print('item_description: ', len(f['item_description'].unique()))
    print('General Info:')
    print(f.info())


    print('value_counts OF THE FEATURES:')
    print(f.item_condition_id.value_counts())
    print(f.shipping.value_counts())
    print(f['brand_name'].value_counts())


    print(f.head())


''' series_to_encode: a 'series' type to be transfered to vectors by infersent '''
# https://github.com/facebookresearch/InferSent
def infersent_encoder(series_to_encode):
    sentences = series_to_encode.tolist()

    nltk.download('punkt')

    V = 2

    MODEL_PATH = '../mercariPriceData/InferSent/encoder/infersent%s.pickle' % V # folders
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))

    W2V_PATH = '../mercariPriceData/dataset/fastText/cc.en.300.vec' # folders
    infersent.set_w2v_path(W2V_PATH)
    # try:
    infersent.build_vocab(sentences, tokenize=True)
    print("done build vocab")
    # except: # DEBUG. this thing is not working for a lot of data...
    #     print(sentences)
    #     print("number 1")
    try:
        embeddings = infersent.encode(sentences, tokenize=True)
        print("done encoding")
        return embeddings
    except:
        print("number 2")




def data_preprocessing():
    data = puredata.copy()

    # for row, val in enumerate(data['item_description']):

    # Change anything with "No description yet" to an empty string..
    for row_index,val in enumerate(data['item_description']):
        if( isinstance(val , str) == False):
            col_index = data.columns.get_loc("item_description")
            print(data.iat[row_index, col_index])
            data.iat[row_index, col_index] = 'No description yet'
            print("after: ", data.iat[row_index, col_index])

    print("done fixing values")
    # ___________________________________________________________________________________________________
    # Using infersent on the item_description column in order to transpose it to vectors (size: 4096)
    data = data.iloc[:2] # DEBUG
    series_descriptions = pd.Series(data["item_description"])
    print(series_descriptions)

    description_embeddings = infersent_encoder(series_descriptions)
    # delete item_description column and add the vectors instead
    data.drop(['item_description'], axis=1)
    data = pd.concat([data, description_embeddings], axis=1)



    # print(data.head(50))
    # TODO: something doesn't work with this. when I run this on the first 5 rows, its ok. when I run it on all of them.
#"TODO:"    it crashes. Need to check for some NaN's or something like this... run on DEBUG more (didn't do it becuase GYM!
    # ___________________________________________________________________________________________________


    return data


if __name__ == '__main__':
    # TODO: tf-idf
    # TODO: figure out how to change the dataframe and save the changes to a CSV, so we can do the preproccessing only once! :)
    data = data_preprocessing()
    #print(data.head())
    #show_data_structure()

    sentences = (data['item_description']).tolist()
    print(sentences)
    # Save training data into a CSV:
    data.to_csv('./numeric_train.csv', encoding='utf_8', index=False)