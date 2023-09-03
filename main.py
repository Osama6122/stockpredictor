import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch as tch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# from normalizer import Normalizer
from dataloader import TimeSeriesDataset
from splitdata import get_train_val_split_data
from splitdata import get_data
from lstm import LSTMModel
from lstm import run_epoch


tch.set_default_dtype(tch.float64)
batch_size = 64
input_size = 1
hidden_layer_size = 128  # Increased hidden layer size
num_layers = 3  # Increased number of layers
dropout = 0.1
device = "cpu"
learning_rate = 0.0005  # Further reduced learning rate
weight_decay = 1e-5  # L2 regularization
step_size = 40
num_epoch = 50  # Increased epochs
window_size = 20
output_size = 1
clip_value = 5  # Gradient clipping value
# split_index = 2507





def main():
    
    d = get_data("IBM")
    data_date = [date for date in d.keys()]
    data_date.reverse()
    num_of_data_points = len(data_date)
    # Prepare data for training and Validation. Data x is input feature data and Data y is corresponding labels

    data_x_train, data_x_val, data_y_train, data_y_val, data_norm_close_price, split_index = get_train_val_split_data()
 
        
    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    # print(data_x_train)
    # print(data_y_train)
    # print(data_y_val)
    # print("dataset train")
    # print(dataset_train.__getitem__(0))

    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    train_dataloader = DataLoader(dataset_train, batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size, shuffle=True)

    my_lstm_model = LSTMModel(input_size, hidden_layer_size, num_layers, output_size, dropout)
    my_lstm_model = my_lstm_model.to(device)
    # print(my_lstm_model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(my_lstm_model.parameters(), learning_rate, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    for epoch in range(num_epoch):
        loss_train, lr_train = run_epoch(my_lstm_model, optimizer, criterion, scheduler, train_dataloader, is_training=True)
        loss_val, lr_val = run_epoch(my_lstm_model, optimizer, criterion, scheduler, val_dataloader)

        tch.nn.utils.clip_grad_norm_(my_lstm_model.parameters(), clip_value)

        scheduler.step(loss_val)

        lr = optimizer.param_groups[0]['lr']
    
        print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
                .format(epoch+1, num_epoch, loss_train, loss_val, lr_train))
    

    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date

    train_dataloader = DataLoader(dataset_train, batch_size, shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size, shuffle=False)

    # my_lstm_model.train()
    my_lstm_model.eval()

    # predict on the training data, to see how well the model managed to learn and memorize

    predicted_train = np.array([])

    for idx, (x, y) in enumerate(train_dataloader):
        # print(f"Before: {x[0]}")
        # i=0
        x = x.to(device)
        # print(f"After: {x[0]}")
        # print(y[0])
        out = my_lstm_model(x)
        # print(out)
        out = out.cpu().detach().numpy()
        # print(out)
        predicted_train = np.concatenate((predicted_train, out))
        
        # print(f"Batch {idx+1} - Date Sequence: {x[i]} - Input Shape: {x.shape}, Output Shape: {out.shape}")
        # print("Predicted Output:")
        # print(out)
        # i = i+1

    

    print(predicted_train)
    print(predicted_train.shape)
    # predict on the validation data, to see how the model does

    predicted_val = np.array([])

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(device)
        out = my_lstm_model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))
        # print(f"Batch {idx+1} - Input Shape: {x.shape}, Output Shape: {out.shape}")
        # print("Predicted Output:")
        print(out)
    
        

    # print(predicted_val)
    # print(predicted_val.shape)
    # prepare data for plotting

    # Initialize to_plot_data_y_train_pred and to_plot_data_y_val_pred as Python lists with None
    to_plot_data_y_train_pred = [None] * num_of_data_points
    to_plot_data_y_val_pred = [None] * num_of_data_points

    # Assign predicted_train and predicted_val to the appropriate slices
    to_plot_data_y_train_pred[window_size: split_index + window_size] = predicted_train.tolist()
    to_plot_data_y_val_pred[split_index + window_size:] = predicted_val.tolist()

    # Convert back to numpy arrays
    to_plot_data_y_train_pred = np.array(to_plot_data_y_train_pred)
    to_plot_data_y_val_pred = np.array(to_plot_data_y_val_pred)
    # plots

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_norm_close_price, label="Actual prices", color="red")
    plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color="blue")
    plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color="green")
    plt.title("Compare predicted prices to actual prices")
    # xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    # x = np.arange(0,len(xticks))
    # plt.xticks(x, xticks, rotation='vertical')
    plt.grid(True)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
