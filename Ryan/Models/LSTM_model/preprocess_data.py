import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# generate multi-class labels
def generate_multi_labels(price_diff, thresholds):
    if thresholds[0] < price_diff <= thresholds[1]:  # stay
        return 3
    elif thresholds[1] < price_diff <= thresholds[3]:  # small up
        return 4
    elif thresholds[3] < price_diff <= thresholds[5]:  # medium up
        return 5
    elif price_diff > thresholds[5]:  # large up
        return 6
    elif thresholds[0] > price_diff >= thresholds[2]:  # small down
        return 2
    elif thresholds[2] > price_diff >= thresholds[4]:  # medium down
        return 1
    elif price_diff < thresholds[4]:  # large down
        return 0


# generate binary labels
def generate_binary_labels(price_diff, thresholds):
    if price_diff > thresholds[1]:  # up
        return 2
    elif price_diff < thresholds[0]:  # down
        return 0
    else:
        return 1  # stay


# def generate_binary_labels(price_diff, thresholds):
#     if price_diff > 1:  # up
#         return 2
#     elif price_diff < -1:  # down
#         return 0
#     else:
#         return 1


# transform time information to cyclical features
def time_to_cyclical(x, period):
    return np.sin(2 * np.pi * x / period), np.cos(2 * np.pi * x / period)


# preprocess the data
# input: data: pandas DataFrame
#        label: 'binary' or 'multi'
# output: data: processed pandas DataFrame,including cyclical features and standardized numerical features
def preprocess(data, label='binary', scale=False):
    # if we only need data from several days:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    # data = data[(data['Datetime'] >= '2025-01-02') & (data['Datetime'] < '2025-01-04')]

    # Calculate the Weighted Average Price
    data['price'] = (data['Max Bid Price'] * data['Max Bid Quantity'] + data['Min Ask Price'] *
                     data['Min Ask Quantity']) / (data['Max Bid Quantity'] + data['Min Ask Quantity'])

    # Calculate the price difference
    data['price_diff'] = data['price'].diff().shift(-1)
    mean_diff = data['price_diff'].mean()  # -0.002
    std_diff = data['price_diff'].std()  # 10.87
    # print('mean_diff:', mean_diff)
    # print('std_diff:', std_diff)

    # create thresholds based on the mean and std of price_diff
    thresholds = [mean_diff - 0.3 * std_diff, mean_diff + 0.3 * std_diff,
                  mean_diff - 1 * std_diff, mean_diff + 1 * std_diff,
                  mean_diff - 2 * std_diff, mean_diff + 2 * std_diff]

    if label == 'binary':
        data['label_binary'] = data['price_diff'].apply(generate_binary_labels, args=(thresholds,))
        data.drop(columns=['price_diff'], inplace=True)
    elif label == 'multi':
        data['label_multi'] = data['price_diff'].apply(generate_multi_labels, args=(thresholds,))
        data.drop(columns=['price_diff'], inplace=True)

    # extract the time information
    data['total_minutes'] = data['Datetime'].dt.hour * 60 + data['Datetime'].dt.minute
    data['dayofweek'] = data['Datetime'].dt.dayofweek
    data['day'] = data['Datetime'].dt.day
    data['month'] = data['Datetime'].dt.month

    # standardize numerical features
    if scale:
        scaler = StandardScaler()
        data[['Max Bid Price', 'Max Bid Quantity', 'Min Ask Price',
              'Min Ask Quantity', 'Total Ask Quantity', 'Total Bid Quantity', 'price']] = scaler.fit_transform(
            data[['Max Bid Price', 'Max Bid Quantity', 'Min Ask Price',
                  'Min Ask Quantity', 'Total Ask Quantity', 'Total Bid Quantity', 'price']])

        # transform the time information to cyclical features
        data['sin_total_minutes'], data['cos_total_minutes'] = time_to_cyclical(data['total_minutes'], 3800)
        data['sin_dayofweek'], data['cos_dayofweek'] = time_to_cyclical(data['dayofweek'], 7)

        # drop the original time information
        data.drop(columns=['total_minutes', 'dayofweek', 'day', 'month'], inplace=True)

    data.drop(columns=['Datetime'], inplace=True)

    return data


if __name__ == '__main__':
    data_path = 'C:\\Users\\yhb\\dsmp-2024-group23\\Ryan\\datasets\\resampled_lob_minALL.csv'
    # data_path='E:\\Bristol\\mini_project\\JPMorgan_Set01\\test_datasets\\resampled_lob_secALL.csv'
    data = pd.read_csv(data_path)

    # 只保留前100000行数据
    # data = data[:500000]

    data = preprocess(data, label='binary', scale=True)
    print(data['label_binary'].value_counts())
    print(data.columns)
    print(data.shape)

    # data.to_csv('C:\\Users\\yhb\\dsmp-2024-group23\\Ryan\\datasets\\resampled_lob_lstm_processed.csv', index=False)
