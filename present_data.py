import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import numpy as np

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

    print("Shipping distribution:")
    print(f.shipping.value_counts() / len(f))

    #print(f.head())


def present_correlated_hist(data):
    
    prc_shipBySeller = data.loc[data.shipping == 1, 'price']
    prc_shipByBuyer = data.loc[data.shipping == 0, 'price']
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.hist(np.log(prc_shipBySeller + 1), color='#8CB4E1', alpha=1.0, bins=50,
            label='Price when Seller pays Shipping')
    ax.hist(np.log(prc_shipByBuyer + 1), color='#007D00', alpha=0.7, bins=50,
            label='Price when Buyer pays Shipping')
    ax.set(title='Histogram Comparison', ylabel='% of Dataset in Bin')
    plt.xlabel('log(price+1)', fontsize=17)
    plt.ylabel('frequency', fontsize=17)
    plt.title('Price Distribution by Shipping Type', fontsize=17)
    plt.tick_params(labelsize=15)
    plt.show()
    plt.clf()
    plt.cla()


def presentations(data):




    # Show information about the price labels
    print(data['price'].describe())
    eps = 0.000001
    log_price_series = np.log(data['price'].radd(1))
    plt.xlabel('prices+1 log-scale')
    plt.ylabel('Number of items with given price')
    log_price_series.hist(bins=50)
    plt.title('prices log-scale')
    plt.show()
    plt.clf()
    plt.cla()

    ''' Showing different numeric values histograms'''
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
