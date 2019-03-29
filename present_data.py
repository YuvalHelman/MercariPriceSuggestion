import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm as cm


def show_data_structure(f):
    """
    A function that prints basic information regarding the data we are dealing with
    """
    print('#################################################')
    print('LOOKING ON THE DATA STRUCTURE:')
    print('#################################################')
    print('data size: ', len(f))


    print('#################################################')
    print("LOOKING ON NUMBER OF UNIQUE VALUES IN EVERY FEATURE:")
    print('#################################################')
    print('item_condition_id: ', len(f['item_condition_id'].unique()))
    print('name: ', len(f['name'].unique()))
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

    #print(f.head())


def presentations(data):

    data['item_condition_id'].hist()
    plt.title('item_condition_id')
    plt.show()
    plt.clf()
    plt.cla()
    #data_p = data.loc[:,['item_condition_id', 'price']]

    # Correlation Matrix of the numeric features
    print(data[['item_condition_id', 'shipping', 'price']].corr())

    data['shipping'].hist()
    plt.title('shipping')
    plt.show()

    # Make a correlation matrix with the item_condition_id , shipping

if __name__ == '__main__':
    puredata = pd.read_csv('./mercariPriceData/dataset/train.tsv', sep='\t', encoding="utf_8")  # change folders

    ''' Show standard information about the Data we're dealing with  '''
    show_data_structure(puredata)
    ''' Show standard histograms etc.  '''
    presentations(puredata)
