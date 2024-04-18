from Ryan.Models.LSTM_model_Regression.preprocess import preprocess_data
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# create sequences for dataloader
# def create_sequences(X, y, seq_length, predict_steps):
#     xs = []
#     ys = []
#     for i in range(len(X) - seq_length - predict_steps + 1):
#         x = X[i:(i + seq_length)]
#         y_seq = y[i + seq_length:i + seq_length + predict_steps]  # 获取输入x后的predict_steps个时间步作为目标
#         xs.append(x)
#         ys.append(y_seq)
#     return np.array(xs), np.array(ys)

def create_sequences(X, y, seq_length, predict_steps):
    xs = []
    ys = []
    for i in range(len(X) - seq_length - predict_steps + 1):
        x = X[i:(i + seq_length)]
        y_seq = y[i + seq_length:i + seq_length + predict_steps]  # 获取输入x后的predict_steps个时间步作为目标
        xs.append(x)
        ys.append(y_seq)

    # Convert ys to have shape (batch_size, predict_steps, output_size)
    ys = np.array(ys)
    ys = np.expand_dims(ys, axis=2)  # Add an axis for output_size
    return np.array(xs), ys


# This function is used to load the data and create the DataLoaders
def load_data(data_path, sequence_length=10, batch_size=64, test_size=0.1, val_size=0.1, predict_steps=1):
    # Load the dataset
    data = preprocess_data(data_path)

    # choose the features and labels
    X = data.drop(columns=['price']).values
    y = data['price'].values

    # Create sequences
    X_seq, y_seq = create_sequences(X, y, sequence_length, predict_steps)

    # Split the data into training, validation, and testing sets
    test_split_idx = int((1 - test_size) * len(X_seq))
    val_split_idx = int((1 - test_size - val_size) * len(X_seq))

    X_train, X_val, X_test = X_seq[:val_split_idx], X_seq[val_split_idx:test_split_idx], X_seq[test_split_idx:]
    y_train, y_val, y_test = y_seq[:val_split_idx], y_seq[val_split_idx:test_split_idx], y_seq[test_split_idx:]

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create dataloaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_data2(data_path, sequence_length=10, batch_size=64, test_size=0.1, val_size=0.1, predict_steps=1):
    # Load the dataset
    train_set, val_set, test_set = preprocess_data(data_path)

    # Assume train_set, val_set, test_set are DataFrames with the same structure
    # X_train = train_set.drop(columns=['price']).values
    X_train = train_set.values
    y_train = train_set['price'].values
    # X_val = val_set.drop(columns=['price']).values
    X_val = val_set.values
    y_val = val_set['price'].values
    # X_test = test_set.drop(columns=['price']).values
    X_test = test_set.values
    y_test = test_set['price'].values

    # Create sequences for each dataset
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length, predict_steps)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length, predict_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length, predict_steps)

    # Convert sequences to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

    # Create DataLoaders 
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader