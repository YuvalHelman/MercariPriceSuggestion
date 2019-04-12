import re
import string
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras import backend as K

train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')
train["source"] = "train"
test["source"] = "test"
train = train.rename(columns={"train_id": "id"})
test = test.rename(columns={"test_id": "id"})
data = pd.concat([train, test])


def preprocessor(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    text = regex.sub(" ", text)  # remove punctuation
    return text


def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        , 'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ)
        , 'brand_name': np.array(dataset.brand_name)
        , 'general_cat': np.array(dataset.general_cat)
        , 'subcat_1': np.array(dataset.subcat_1)
        , 'subcat_2': np.array(dataset.subcat_2)
        , 'item_condition': np.array(dataset.item_condition_id)
        , 'shipping': np.array(dataset.shipping)
        , 'len_desc': np.array(dataset.len_description)
        , 'len_name': np.array(dataset.len_name)
    }
    return X


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def get_model():
    # params
    dr_r = 0.1

    # Inputs
    name = Input(shape=[MAX_NAME_SEQ], name="name")
    item_desc = Input(shape=[MAX_ITEM_DESC_SEQ], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    general_cat = Input(shape=[1], name="general_cat")
    subcat_1 = Input(shape=[1], name="subcat_1")
    subcat_2 = Input(shape=[1], name="subcat_2")
    item_condition = Input(shape=[1], name="item_condition")
    shipping = Input(shape=[1], name="shipping")
    len_desc = Input(shape=[1], name="len_desc")
    len_name = Input(shape=[1], name="len_name")

    # Embeddings layers
    emb_name = Embedding(MAX_NAME, 50)(name)
    emb_item_desc = Embedding(MAX_DESC, 50)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_general_cat = Embedding(MAX_GENERAL_CAT, MAX_GENERAL_CAT)(general_cat)
    emb_subcat_1 = Embedding(MAX_SUBCAT_1, 20)(subcat_1)
    emb_subcat_2 = Embedding(MAX_SUBCAT_2, 20)(subcat_2)
    emb_item_condition = Embedding(MAX_CONDITION, MAX_CONDITION)(item_condition)

    # rnn layer
    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_name)

    # main layer
    main_l = concatenate([
        Flatten()(emb_brand_name)
        , Flatten()(emb_general_cat)
        , Flatten()(emb_subcat_1)
        , Flatten()(emb_subcat_2)
        , Flatten()(emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , shipping
        , len_desc
        , len_name
    ])
    main_l = Dropout(dr_r)(Dense(512)(main_l))
    main_l = Dropout(dr_r)(Dense(256)(main_l))
    main_l = Dropout(dr_r)(Dense(128)(main_l))
    main_l = Dropout(dr_r)(Dense(64)(main_l))

    # output
    output = Dense(1, activation="linear")(main_l)

    # model
    model = Model([name, item_desc, brand_name
                      , general_cat, subcat_1, subcat_2, item_condition, shipping, len_desc, len_name], output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae", root_mean_squared_error])

    return model


if __name__ == '__main__':

    data['general_cat'], data['subcat_1'], data['subcat_2'] = \
        zip(*data['category_name'].apply(lambda x: split_cat(x)))
    data.drop('category_name', axis=1, inplace=True)
    data.brand_name = data.brand_name.fillna("None")

    le = LabelEncoder()
    labels = ["brand_name", "general_cat", "subcat_1", "subcat_2"]
    for label in labels:
        data[label] = data[label].astype("str")
        le.fit(np.hstack(data[label]))
        data[label] = le.transform(data[label])

    data.head()

    data.item_description = data.item_description.fillna("None")
    data.name = data.name.fillna("None")

    data.item_description = data.item_description.apply(lambda x: preprocessor(x))
    data.name = data.name.apply(lambda x: preprocessor(x))

    raw_text = np.hstack([data[data.source == "train"].item_description.str.lower()])

    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)

    data["seq_item_description"] = tok_raw.texts_to_sequences(data.item_description.str.lower())
    MAX_DESC = len(tok_raw.word_counts) + 1
    raw_text = np.hstack([data[data.source == "train"].name.str.lower()])

    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)

    data["seq_name"] = tok_raw.texts_to_sequences(data.name.str.lower())
    MAX_NAME = len(tok_raw.word_counts) + 1
    data["len_description"] = data.seq_item_description.apply(lambda x: len(x))
    data["len_name"] = data.seq_name.apply(lambda x: len(x))
    data.drop("name", axis=1, inplace=True)
    data.drop("item_description", axis=1, inplace=True)
    data.len_description.describe()
    train = data[data.source == "train"]
    test = data[data.source == "test"]
    train.head()

    MAX_NAME_SEQ = 10
    MAX_ITEM_DESC_SEQ = 50
    MAX_GENERAL_CAT = data.general_cat.max() + 1
    MAX_SUBCAT_1 = data.subcat_1.max() + 1
    MAX_SUBCAT_2 = data.subcat_2.max() + 1
    MAX_BRAND = data.brand_name.max() + 1
    MAX_CONDITION = data.item_condition_id.max() + 1
    dtrain, dvalid = train_test_split(train, train_size=0.99, random_state=123)

    X_train = get_keras_data(dtrain)
    X_valid = get_keras_data(dvalid)
    X_test = get_keras_data(test)

    model = get_model()
    model.summary()
    BATCH_SIZE = 10000
    epochs = 5

    model = get_model()
    model.fit(X_train, np.log1p(dtrain.price), epochs=epochs, batch_size=BATCH_SIZE
              , validation_data=(X_valid, np.log1p(dvalid.price))
              , verbose=1)

    val_preds = model.predict(X_valid)
    score = np.sqrt(mean_squared_error(np.log1p(dvalid.price), np.array(val_preds)))
    print("Score: " + str(score))
