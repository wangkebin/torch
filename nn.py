import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch as torch
import torch.nn as nn
import torch.optim as optim

from data_loader import iris_dataloader

#init 

class neuNet(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1,hidden_dim2 )
        self.layer3 = nn.Linear(hidden_dim2, out_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# training set, validation set, and test set
custom_dateset = iris_dataloader("./iris/iris.data")
training_size = int(len(custom_dateset) * 0.7)
val_size = int(len(custom_dateset) * 0.2)
test_size = len(custom_dateset) - training_size - val_size

ts, vs, tts = torch.utils.data.random_split(custom_dateset, [training_size, val_size, test_size])

tl = DataLoader(ts, batch_size=16, shuffle=False)
vl = DataLoader(vs, batch_size=1, shuffle=False)
tl = DataLoader(ts, batch_size=1, shuffle=False)

