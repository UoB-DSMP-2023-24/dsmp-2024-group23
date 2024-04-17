from torch import nn
import torch
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

        # 初始化隐藏状态和细胞状态
        self.hidden = None

    def reset_hidden_state(self):
        self.hidden = None  # 重置状态，可以在需要时调用

    def forward(self, x):
        if self.hidden is None:
            # 初始化状态
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            self.hidden = (h0, c0)
        else:
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())  # 分离隐藏状态，防止梯度回传历史

        out, self.hidden = self.lstm(x, self.hidden)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        return out

# for epoch in range(num_epochs):
#     model.reset_hidden_state()  # 在每个epoch开始时重置隐藏状态,



class LSTMModel_multi(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2, predict_steps=1):
        super(LSTMModel_multi, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_steps = predict_steps  # 控制输出多少时间步的预测

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        # 修改输出层，直接输出多步预测结果
        self.fc = nn.Linear(hidden_size, output_size * predict_steps)
        self.hidden = None

    def reset_hidden_state(self):
        self.hidden = None

    def forward(self, x):
        if self.hidden is None or self.hidden[0].size(1) != x.size(0):
            # 重新初始化隐藏状态以匹配当前批次大小
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            self.hidden = (h0, c0)
        else:
            # 分离隐藏状态，这对于避免梯度爆炸很有必要
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        out, self.hidden = self.lstm(x, self.hidden)
        out = self.dropout(out[:, -1, :])  # 只取序列的最后一个输出
        out = self.fc(out)
        out = out.view(x.size(0), self.predict_steps, -1)  # 重新调整输出尺寸以匹配预测步数
        return out



if __name__ == '__main__':
    model=LSTMModel(11, 256, 2, 1, 0.2)
    print(model)