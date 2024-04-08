import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Ryan.Models.LSTM_model.run import plot_metrics, train, validate, test, EarlyStopping
from Ryan.Models.LSTM_model.run_steps import train_steps, validate_steps, test_steps
from dataloader import load_data
from lstm_model import LSTMModel, LSTMModel_dropout, init_weights_orthogonal, LSTMModel_dropout_steps, LSTMModel_dropout_steps, EnhancedLSTMModel
from torch import nn, optim

if __name__ == '__main__':
    data_path='C:\\Users\\yhb\\dsmp-2024-group23\\Ryan\\datasets\\resampled_lob_minALL.csv'
    # data_path = 'E:\\Bristol\\mini_project\\JPMorgan_Set01\\test_datasets\\resampled_lob_secALL.csv'

    train_loader, test_loader,val_loader = load_data(data_path, sequence_length=20,
                                          batch_size=64, test_size=0.1,val_size=0.1, scale=True,label='binary',sequence_loss=True)

    print('Train Loader:', len(train_loader.dataset))
    print('Test Loader:', len(test_loader.dataset))
    print('Validation Loader:', len(val_loader.dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel_dropout_steps(input_size=11, hidden_size=100, num_layers=1, fc_size=50, output_size=3,dropout_rate=0).to(device)
    # model.apply(init_weights_orthogonal)  # 初始化权重: 正交初始化

    # optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)  # Adam优化器, weight_decay表示L2正则化
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

    # 三分类，设置类别权重
    # label[0,1,2]分别对应2:1:1的比例，则设置权重为[0.5,1,1]
    # class_weights = torch.tensor([1,0.1,1]).to(device)
    # criterion_train = nn.CrossEntropyLoss(weight=class_weights)

    criterion_train = nn.CrossEntropyLoss()  # 训练集的损失函数
    criterion_val = nn.CrossEntropyLoss() # 验证集和测试集的损失不需要考虑权重

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)  # 学习率调度器:监控验证集上的损失,动态调整学习率
    # early_stopping = EarlyStopping(patience=10, min_delta=0.01)  # 早停器

    train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies = [], [], [], [], [], []
    # loss_list = []

    for epoch in range(1, 200):
        # train(model, device, train_loader, optimizer, epoch, train_losses, train_accuracies)
        train_steps(model, device, train_loader, optimizer, epoch, train_losses, train_accuracies, criterion_train)
        val_loss = validate_steps(model, device, val_loader, val_losses, val_accuracies, criterion_val)
        # test(model, device, test_loader, test_losses, test_accuracies)  # 在每个epoch后执行验证

        scheduler.step(val_loss) # 根据验证集的损失调整学习率

        # 保存模型的条件可以根据需要自定义，例如每隔一定数量的epoch保存一次

        if epoch % 50 == 0:  # 每10个epoch保存一次模型
            save_path = f'saved_models/model_epoch_{epoch}.pt'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        # if epoch % 10 == 0 and epoch > 50:
        #     plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies)

        # 当验证集的损失不再下降时，停止训练
        # early_stopping(val_loss)
        # if early_stopping.early_stop:
        #     print("Early stopping triggered")
        #     break

    # 所有训练完成后，再进行一次测试
    test(model, device, test_loader, test_losses, test_accuracies, criterion_val)

    plot_metrics(train_losses,train_accuracies,val_losses,val_accuracies)

