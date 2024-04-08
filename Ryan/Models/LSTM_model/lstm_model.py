from torch import nn
import torch
import torch.nn.functional as F


# 定义LSTM模型
# 基础LSTM模型，只有一个lstm层和一个全连接层，输出为最后一个时间步的输出，且没有使用softmax，便于直接使用CrossEntropyLoss
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM循环单元
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


# Dropout LSTM
class LSTMModel_dropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel_dropout, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM循环单元
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        return out


# 两个全连接层，无dropout，使用relu激活函数对第一个全连接层的输出进行激活，添加非线性
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, fc_size, output_size):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM循环单元
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 第一个全连接层
        self.fc1 = nn.Linear(hidden_size, fc_size)
        # 第二个全连接层，作为解码层
        self.fc2 = nn.Linear(fc_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]

        out = F.relu(self.fc1(out))

        # 通过第二个全连接层（解码层）
        out = self.fc2(out)

        return out


# 两个全连接层，增加dropout
# fc_size一般介于hidden_size和output_size之间
class EnhancedLSTM_dropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, fc_size, output_size, dropout_rate=0.2):
        super(EnhancedLSTM_dropout, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM循环单元
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


# keep output of all time steps
class LSTMModel_dropout_steps(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, fc_size=50,dropout_rate=0.5):
        super(LSTMModel_dropout_steps, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate) # 如果dropout_rate=0，则不会应用dropout
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        # 确保全连接层被应用于所有时间步
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out


# 正交初始化
def init_weights_orthogonal(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
