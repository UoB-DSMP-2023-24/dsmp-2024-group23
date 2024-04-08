import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from Ryan.Models.LSTM_model.preprocess_data import preprocess


# This function is used to load the data and create the DataLoaders
# first, read the data from the csv file
# then preprocess the data via the preprocess function
# then create sequences of the data, which is used as the input features and labels
# finally, split the data into training and testing sets, and create DataLoaders

# input: data_path: the path of the csv file
#        sequence_length: the length of the sequence
#        batch_size: the batch size
#        test_size: the size of the test set
#        val_size: the size of the validation set,if None, no validation set will be created
#        scale: whether to scale the numerical features,default is True

## not shuffle any sequences
# def load_data(data_path, sequence_length=10, batch_size=64, test_size=0.2, val_size=None, scale=True):
#     # Load the dataset
#     data = pd.read_csv(data_path)
#     data = preprocess(data, label='binary', scale=scale)
#
#     # Drop the 'Datetime' column and separate features from labels
#     X = data.drop(columns=['label_binary']).values
#     y = data['label_binary'].values
#
#     # Function to create sequences
#     def create_sequences(data, seq_length):
#         xs = []
#         ys = []
#         for i in range(len(data) - seq_length):
#             x = data[i:(i + seq_length)]
#             xs.append(x)  # 10 time steps for input features
#             ys.append(y[i + seq_length])  # the next time step for the label
#         return np.array(xs), np.array(ys)
#
#     # 创建序列
#     X_seq, y_seq = create_sequences(X, sequence_length)
#
#     if val_size is not None:
#         # 先从全部数据中划分出训练数据和剩余数据（测试集和验证集的合集）
#         X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=(test_size + val_size), shuffle=False)
#
#         # 再从剩余数据中划分出测试集和验证集
#         test_size_adjusted = test_size / (test_size + val_size)  # 调整测试集的比例
#         X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size_adjusted, shuffle=False)
#     else:
#         # 直接从全部数据中划分出训练集和测试集
#         X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=test_size, shuffle=False)
#         X_val, y_val = None, None
#
#     # 转换为PyTorch张量
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#     y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
#
#     # 创建TensorDatasets
#     train_data = TensorDataset(X_train_tensor, y_train_tensor)
#     test_data = TensorDataset(X_test_tensor, y_test_tensor)
#
#     # 创建DataLoaders
#     train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
#     test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
#
#     # 对于验证集的处理
#     if val_size is not None:
#         X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
#         y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
#         val_data = TensorDataset(X_val_tensor, y_val_tensor)
#         val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
#         return train_loader, test_loader, val_loader
#     else:
#         return train_loader, test_loader

# shuffle the training sequences (keeping the order inside each sequence) before creating the DataLoader
def load_data(data_path, sequence_length=10, batch_size=64, test_size=0.2, val_size=None, scale=True, label='binary',
              sequence_loss=False):
    # Load the dataset
    data = pd.read_csv(data_path)
    # data=data[:100000]
    data = preprocess(data, label=label, scale=scale)

    # Drop the 'Datetime' column and separate features from labels
    if label == 'binary':
        X = data.drop(columns=['label_binary']).values
        y = data['label_binary'].values
    elif label == 'multi':
        X = data.drop(columns=['label_multi']).values
        y = data['label_multi'].values

    # Function to create sequences
    # default method (calculate loss on batches)

    def create_sequences(data, seq_length):
        xs = []
        ys = []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            xs.append(x)  # 10 time steps for input features
            ys.append(y[i + seq_length])  # the next time step for the label
        return np.array(xs), np.array(ys)

    # # calculate loss on all time steps
    def create_sequences_steps(data, labels, seq_length):
        xs = []
        ys = []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = labels[i + 1:i + 1 + seq_length]  # 获取与输入x对应的标签序列
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    if sequence_loss: # 生成序列标签
        X_seq, y_seq = create_sequences_steps(X, y, sequence_length)
    else: # 每个序列的最后一个时间步作为标签
        X_seq, y_seq = create_sequences(X, sequence_length)

    # print(f"X_seq shape: {X_seq.shape}")
    # print(f"y_seq shape: {y_seq.shape}")
    # IF sequence_loss is True, y_seq will be a 3D array, otherwise a 2D array

    # 分割数据集
    test_split_size = test_size if val_size is None else test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=test_split_size, shuffle=False)

    # 打乱训练集的序列间顺序
    train_indices = np.arange(X_train.shape[0])
    np.random.shuffle(train_indices)
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    if val_size is not None:
        # 从剩余数据中分割出验证集和测试集
        val_size_adjusted = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - val_size_adjusted, shuffle=False)
    else:
        X_val, y_val = None, None
        X_test, y_test = X_temp, y_temp

    # 转换为PyTorch张量并创建DataLoaders
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)  # 已经预先打乱

    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    if val_size is not None:
        val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader
    else:
        return train_loader, test_loader


if __name__ == '__main__':
    data_path = 'C:\\Users\\yhb\\dsmp-2024-group23\\Ryan\\datasets\\resampled_lob_minALL.csv'
    # data_path = 'E:\\Bristol\\mini_project\\JPMorgan_Set01\\test_datasets\\resampled_lob_secALL.csv'
    train_loader, test_loader, val_loader = load_data(data_path, sequence_length=30, val_size=0.1, scale=True,
                                                      label='multi', sequence_loss=False)
    print(len(train_loader), len(test_loader), len(val_loader))
