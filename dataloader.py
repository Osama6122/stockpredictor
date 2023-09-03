import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self,x,y):

        float_array = []
        for sequence in x:
            float_array.append([0]*20)
            for idx in range(len(sequence)):
                float_array[-1][idx] = float(sequence[idx].replace("-", ""))

        arr = np.array(float_array)
        arr = np.expand_dims(arr, 2) 
        self.y = y
        self.x = arr

        # print(arr[0])
        
        print(self.x.dtype)
        print(self.y.dtype)


    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


