import numpy as np
import json
import matplotlib.pyplot as plt
from normalizer import Normalizer


def get_data(input_file):
    with open(f"./temp/{input_file}.json" , "r") as f:
        data = json.load(f)
    f.close()
    return data


def prepare_data_x(x, size):
    num_of_rows = x.shape[0] - size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(num_of_rows, size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]
    

def prepare_data_y(x, window_size):
    output = x[window_size:]
    return output


def get_train_val_split_data():

    window_size = 20  # LSTM Window/Sequence size
    split_ratio = 0.5 # Data split ratio

    d = get_data("IBM")
    data_date = [date for date in d.keys()]
    data_close_price = [float(values["5. adjusted close"]) for values in d.values()]
    data_date.reverse()
    data_close_price.reverse()

    normalizer = Normalizer()
    data_norm_close_price = normalizer.fit_transform(data_close_price)

    num_of_data_points = len(data_date)

    close_price_arr = np.array(data_norm_close_price)
    date_arr = np.array(data_date)
    

    # Prepare data for training and Validation. Data x is input feature data and Data y is corresponding labels
    data_x, data_x_unseen = prepare_data_x(date_arr, window_size)
    data_y = prepare_data_y(close_price_arr, window_size)

    # Splits the data according to split ratio
    split_index = int(data_y.shape[0] * split_ratio)

    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    return data_x_train, data_x_val, data_y_train, data_y_val, data_norm_close_price, split_index


get_train_val_split_data()



