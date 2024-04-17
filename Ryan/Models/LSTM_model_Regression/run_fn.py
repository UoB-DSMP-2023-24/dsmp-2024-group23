import torch
from transformers import get_linear_schedule_with_warmup
import tqdm
from Ryan.Models.LSTM_model_Regression.load_data import load_data, load_data2
from Ryan.Models.LSTM_model_Regression.lstm_models import LSTMModel_multi, LSTMModel


# create empty train_losses=[]
# create training function
def train_fn(model, device, train_loader, optimizer, criterion, scheduler, epoch, train_losses, mae):
    model.train()
    train_loss = 0
    total_abs_error = 0

    # for batch_idx, (data, target) in enumerate(train_loader):
    for data, target in tqdm.tqdm(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        total_abs_error += torch.sum(torch.abs(output - target)).item()

    average_loss = train_loss / len(train_loader)
    train_losses.append(average_loss)

    average_mae = total_abs_error / len(train_loader.dataset)
    mae.append(average_mae)

    print(f"Epoch {epoch}, Training Loss: {average_loss}, Training MAE: {average_mae}")

    return train_losses, mae


def eval_fn(model, device, val_loader, criterion, val_losses, mae, mode="Validation"):
    model.eval()
    val_loss = 0
    total_abs_error = 0
    predictions = []  # 初始化存储预测值的列表
    actuals = []      # 初始化存储实际值的列表

    with torch.no_grad():
        for data, target in tqdm.tqdm(val_loader):
            data, target = data.to(device), target.to(device)
            # data: [batch_size, seq_length, input_size]
            # target: [batch_size, output_size]

            output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            total_abs_error += torch.sum(torch.abs(output - target)).item()

            if mode == "Test":  # 仅当模式为测试时收集预测和实际值
                predictions.append(output.detach().cpu().numpy())
                actuals.append(target.detach().cpu().numpy())

    average_loss = val_loss / len(val_loader)
    val_losses.append(average_loss)

    average_mae = total_abs_error / len(val_loader.dataset)
    mae.append(average_mae)

    print(f"{mode} Loss: {average_loss:.4f}, MAE: {average_mae:.4f}")

    if mode == "Test":
        return val_losses, mae, predictions, actuals
    else:
        return val_losses, mae


def run_model():
    data_path = 'E:\\Bristol\\mini_project\\JPMorgan_Set01\\test_datasets\\resampled_lob_secALL.csv'
    sequence_length = 20
    batch_size = 64
    epochs = 2
    test_size = 0.1
    val_size = 0.1

    input_size =12
    hidden_size = 256
    num_layers = 2
    output_size = 1
    dropout_rate = 0.2
    predict_steps = 1

    train_loader, val_loader, test_loader = load_data2(data_path, sequence_length, batch_size, test_size, val_size,
                                                      predict_steps)
    print('Data loaded successfully')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel_multi(input_size, hidden_size, num_layers, output_size, dropout_rate, predict_steps).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer=torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.MSELoss()

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    train_losses, val_losses, train_mae, val_mae = [], [], [], []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}")
        model.reset_hidden_state() # reset the hidden state for each epoch
        train_losses,train_mae = train_fn(model, device, train_loader, optimizer, criterion, scheduler, epoch, train_losses,
                                train_mae)
        val_losses,val_mae = eval_fn(model, device, val_loader, criterion, val_losses, val_mae, mode="Validation")

        # every 50 epochs save the model
        if epoch % 10 == 0:
            save_path = f'saved_models_regression/model_epoch_{epoch}.pt'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # test the model
    test_losses, test_mae = [], []
    test_losses, test_mae, test_predictions, test_actuals = eval_fn(model, device, test_loader, criterion, test_losses,
                                                                    test_mae, mode="Test")

    return train_losses, val_losses, train_mae, val_mae, test_losses, test_mae, test_predictions, test_actuals

if __name__ == '__main__':
    train_losses, val_losses, train_mae, val_mae, test_losses, test_mae, test_predictions, test_actuals = run_model()

