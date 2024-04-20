import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


# This function is used to cycle the time features
def time_to_cyclical(x, period):
    return np.sin(2 * np.pi * x / period), np.cos(2 * np.pi * x / period)


# preprocess the data
# input: data: data path
# output: data: processed pandas DataFrame,including cyclical features and standardized numerical features
# def preprocess_data(data_path):
#     df = pd.read_csv(data_path)
#     df.dropna(inplace=True)
#     # df['price'] = (df['Min Ask Price'] * df['Min Ask Quantity'] + df['Max Bid Price'] * df['Max Bid Quantity']) / (
#                 # df['Min Ask Quantity'] + df['Max Bid Quantity'])
#
#     # time features
#     df['Datetime'] = pd.to_datetime(df['Datetime'])
#     df['day_of_week'] = df['Datetime'].dt.dayofweek
#     df['month'] = df['Datetime'].dt.month
#     df['day'] = df['Datetime'].dt.day
#     df['hour'] = df['Datetime'].dt.hour
#     df['minute'] = df['Datetime'].dt.minute
#     df['second'] = df['Datetime'].dt.second
#
#     # 将hour+minute+second转换为second
#     df['time_seconds'] = df['hour'] * 3600 + df['minute'] * 60 + df['second']
#
#     # scale the features
#     # scaler = MinMaxScaler()
#     # df[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
#     #     'Total Bid Quantity', 'Total Ask Quantity']] = (
#     #     scaler.fit_transform(df[['Min Ask Quantity', 'Max Bid Quantity', 'price',
#     #                              'time_seconds', 'Total Bid Quantity', 'Total Ask Quantity']]))
#
#     # cosine and sine transformation for day and month
#     df['day_sin'], df['day_cos'] = time_to_cyclical(df['day'], 31)
#     df['month_sin'], df['month_cos'] = time_to_cyclical(df['month'], 12)
#     df['day_of_week_sin'], df['day_of_week_cos'] = time_to_cyclical(df['day_of_week'], 7)
#
#     # choose the features
#     features = ['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
#                 'Total Bid Quantity', 'Total Ask Quantity', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
#                 'day_of_week_sin', 'day_of_week_cos']
#
#     # return the new dataframe
#     df = df[features]
#
#     # split the data
#     train_set, test_set = train_test_split(df, test_size=0.1, shuffle=False)
#     train_set, val_set = train_test_split(train_set, test_size=0.1, shuffle=False)
#
#     # scale the features
#     scaler = MinMaxScaler()
#     train_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
#                'Total Bid Quantity', 'Total Ask Quantity']] = (
#         scaler.fit_transform(train_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
#                                        'Total Bid Quantity', 'Total Ask Quantity']]))
#     val_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
#                 'Total Bid Quantity', 'Total Ask Quantity']] = (
#             scaler.transform(val_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
#                                     'Total Bid Quantity', 'Total Ask Quantity']]))
#     test_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
#                 'Total Bid Quantity', 'Total Ask Quantity']] = (
#             scaler.transform(test_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
#                                     'Total Bid Quantity', 'Total Ask Quantity']]))
#
#     # train_set=train_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
#     #            'Total Bid Quantity', 'Total Ask Quantity']]
#     # val_set=val_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
#     #            'Total Bid Quantity', 'Total Ask Quantity']]
#     # test_set=test_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
#     #               'Total Bid Quantity', 'Total Ask Quantity']]
#
#     return train_set, val_set, test_set


def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    # df['price'] = (df['Min Ask Price'] * df['Min Ask Quantity'] + df['Max Bid Price'] * df['Max Bid Quantity']) / (
    # df['Min Ask Quantity'] + df['Max Bid Quantity'])

    # time features
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['day_of_week'] = df['Datetime'].dt.dayofweek
    df['month'] = df['Datetime'].dt.month
    df['day'] = df['Datetime'].dt.day
    df['hour'] = df['Datetime'].dt.hour
    df['minute'] = df['Datetime'].dt.minute
    df['second'] = df['Datetime'].dt.second

    # 将hour+minute+second转换为second
    df['time_seconds'] = df['hour'] * 3600 + df['minute'] * 60 + df['second']

    # scale the features
    # scaler = MinMaxScaler()
    # df[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
    #     'Total Bid Quantity', 'Total Ask Quantity']] = (
    #     scaler.fit_transform(df[['Min Ask Quantity', 'Max Bid Quantity', 'price',
    #                              'time_seconds', 'Total Bid Quantity', 'Total Ask Quantity']]))

    # cosine and sine transformation for day and month
    df['day_sin'], df['day_cos'] = time_to_cyclical(df['day'], 31)
    df['month_sin'], df['month_cos'] = time_to_cyclical(df['month'], 12)
    df['day_of_week_sin'], df['day_of_week_cos'] = time_to_cyclical(df['day_of_week'], 7)

    # choose the features
    features = ['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
                'Total Bid Quantity', 'Total Ask Quantity', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'day_of_week_sin', 'day_of_week_cos', 'ofi', 'RSI', 'MA', 'Momentum']

    # return the new dataframe
    df = df[features]

    # split the data
    train_set, test_set = train_test_split(df, test_size=0.1, shuffle=False)
    train_set, val_set = train_test_split(train_set, test_size=0.1, shuffle=False)

    # scale the features
    scaler = MinMaxScaler()
    # scaler= StandardScaler()
    train_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
               'Total Bid Quantity', 'Total Ask Quantity', 'ofi', 'RSI', 'MA', 'Momentum']] = (
        scaler.fit_transform(train_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
                                        'Total Bid Quantity', 'Total Ask Quantity', 'ofi', 'RSI', 'MA', 'Momentum']]))
    val_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
             'Total Bid Quantity', 'Total Ask Quantity', 'ofi', 'RSI', 'MA', 'Momentum']] = (
        scaler.transform(val_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
                                  'Total Bid Quantity', 'Total Ask Quantity', 'ofi', 'RSI', 'MA', 'Momentum']]))
    test_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
              'Total Bid Quantity', 'Total Ask Quantity', 'ofi', 'RSI', 'MA', 'Momentum']] = (
        scaler.transform(test_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
                                   'Total Bid Quantity', 'Total Ask Quantity', 'ofi', 'RSI', 'MA', 'Momentum']]))

    # train_set=train_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
    #            'Total Bid Quantity', 'Total Ask Quantity']]
    # val_set=val_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
    #            'Total Bid Quantity', 'Total Ask Quantity']]
    # test_set=test_set[['Min Ask Quantity', 'Max Bid Quantity', 'price', 'time_seconds',
    #               'Total Bid Quantity', 'Total Ask Quantity']]

    return train_set, val_set, test_set

# example usage
# if __name__ == '__main__':
#     data_path = 'E:\\Bristol\\mini_project\\JPMorgan_Set01\\test_datasets\\resampled_lob_secALL.csv'
#     data = preprocess_data(data_path)
#     print(data.head())
