# Final Project regarding the following Kaggle competition:
# https://www.kaggle.com/c/mercari-price-suggestion-challenge

# Submitted by Yuval Helman , Jakov Zingerman

import fasttext
from InferSent.encoder.models import InferSent
import pandas as pd
import torch

def show_data_structure(f):

    # Looking on the data structure:
    print('data size: ', len(f))
    print(f['name'].unique())

    print( f.item_condition_id.value_counts() )

    print(len(f['item_condition_id'].unique()))
    print(len(f['category_name'].unique()))
    print(len(f['brand_name'].unique()))
    print(len(f['price'].unique()))
    print(len(f['shipping'].unique()))
    print(len(f['item_description'].unique()))
    print(f.info())


    print('value_counts of condition of the item:')
    print(f.item_condition_id.value_counts())
    print(f.shipping.value_counts())

    print(f.info())

    data['brand_name'].value_counts()

    print(data.head())

def playing_with_infersent():

    model_version = 1
    MODEL_PATH = "../encoder/infersent%s.pkl" % model_version
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))



# # loading the
# import io
#
# def load_vectors(fname):
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = map(float, tokens[1:])
#     return data
#
# vectors = pd.read_csv(r'/content/drive/My Drive/Colab Notebooks/data/cc.en.300.vec', sep='\t' , encoding="latin1")

if __name__ == '__main__'

    data = pd.read_csv(r'/content/drive/My Drive/Colab Notebooks/data/train.tsv', sep='\t', encoding="latin1")

    show_data_structure(data)
    playing_with_infersent(data)

