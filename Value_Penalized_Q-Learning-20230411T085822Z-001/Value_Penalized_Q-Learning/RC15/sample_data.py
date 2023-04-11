import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utility import to_pickled_df



if __name__ == '__main__':
    data_directory = 'data'
    click_df = pd.read_csv(os.path.join(data_directory, 'yoochoose-clicks.dat'), header=None)
    click_df.columns = ['session_id', 'timestamp', 'item_id','category']

    # filter out the items that interacted less than 3 times
    print('before item filter out , len.click', len(click_df))
    click_df['valid_item'] = click_df.item_id.map(click_df.groupby('item_id')['session_id'].size() > 2)
    click_df = click_df.loc[click_df.valid_item].drop('valid_item', axis=1)
    print('after item filter out,  len.click', len(click_df))

    # flitering out the sessions that len < 3
    click_df['valid_session'] = click_df.session_id.map(click_df.groupby('session_id')['item_id'].size() > 2) # raw dataset > 2, ---> 5
    click_df = click_df.loc[click_df.valid_session].drop('valid_session', axis=1)


    temp = click_df.groupby('session_id')['item_id'].unique().apply(lambda x: len(x))
    print('unique len mean  ', np.mean(temp))
    print('unique len std   ', np.std(temp))

    print('the averaged len of sessions is ', np.mean(click_df.groupby('session_id')['item_id'].size()))
    print('and the std is                  ', np.std(click_df.groupby('session_id')['item_id'].size()))

    buy_df = pd.read_csv(os.path.join(data_directory, 'yoochoose-buys.dat'), header=None)
    buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

    sampled_num = len(click_df.session_id.unique())
    print('total session id', sampled_num)
    # 4,431,931

    sampled_session_id = np.random.choice(click_df.session_id.unique(), 200000, replace=False)
    sampled_click_df = click_df.loc[click_df.session_id.isin(sampled_session_id)]
    sampled_buy_df = buy_df.loc[buy_df.session_id.isin(sampled_click_df.session_id)]
    print('num of sampled click and buy df ', len(sampled_click_df), len(sampled_buy_df))
    print('range of sampled click and buy df ', min(sampled_click_df.item_id), max(sampled_click_df.item_id), min(sampled_buy_df.item_id), max(sampled_buy_df.item_id))

    # encoder the item id
    item_encoder = LabelEncoder()
    merged_df = list(sampled_click_df.item_id) + list(sampled_buy_df.item_id)
    encoded_merged_df = item_encoder.fit_transform(merged_df)

    sampled_click_df['item_id'] = encoded_merged_df[:len(sampled_click_df)]
    sampled_buy_df['item_id'] = encoded_merged_df[len(sampled_click_df):]
    print('num of sampled click and buy df ', len(sampled_click_df), len(sampled_buy_df))
    print('range of sampled click and buy df ', min(sampled_click_df.item_id), max(sampled_click_df.item_id), min(sampled_buy_df.item_id), max(sampled_buy_df.item_id))

    # the wrong way:
    # sampled_click_df['item_id'] = item_encoder.fit_transform(sampled_click_df.item_id)
    # sampled_buy_df['item_id'] = item_encoder.transform(sampled_buy_df.item_id)
    '''
    >>> from sklearn.preprocessing import LabelEncoder
    >>> ls = [1,2,3,3,3]
    >>> lr = [2,2,6,6,6]
    >>> encoder = LabelEncoder()
    >>> print(encoder.fit_transform(ls))
    [0 1 2 2 2]
    >>> print(encoder.fit_transform(lr))
    [0 0 1 1 1]
    '''
    to_pickled_df(data_directory,sampled_clicks=sampled_click_df)
    to_pickled_df(data_directory,sampled_buys=sampled_buy_df)