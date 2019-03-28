import matplotlib.pyplot as plt
import pandas as pd


def show_data_structure(f):
    """
    A function that prints basic information regarding the data we are dealing with
    """
    print('#################################################')
    print('LOOKING ON THE DATA STRUCTURE:')
    print('#################################################')
    print('data size: ', len(f))
    print(f['name'].unique())

    print('#################################################')
    print("LOOKING ON NUMBER OF UNIQUE VALUES IN EVERY FEATURE:")
    print('#################################################')
    print('item_condition_id: ', len(f['item_condition_id'].unique()))
    print('category_name: ', len(f['category_name'].unique()))
    print('brand_name: ', len(f['brand_name'].unique()))
    print('price: ', len(f['price'].unique()))
    print('shipping: ', len(f['shipping'].unique()))
    print('item_description: ', len(f['item_description'].unique()))
    print('General Info:')
    print(f.info())

    print('#################################################')
    print('value_counts OF THE FEATURES:')
    print('#################################################')
    print(f.item_condition_id.value_counts())
    print(f.shipping.value_counts())
    print(f['brand_name'].value_counts())


    for column in list(f.columns.values):
        data['price'].hist()
        plt.show()

    print(f.head())


def presentations(data):

    for column in list(data.columns.values):
        data['price'].hist()
        plt.show()



if __name__ == '__main__':
    puredata = pd.read_csv('./mercariPriceData/dataset/train.tsv', sep='\t', encoding="utf_8")  # change folders

    print(puredata.info())



    presentations(puredata)