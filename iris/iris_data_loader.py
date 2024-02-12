from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch

class iris_dataloader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

        assert os.path.exists(self.data_path), "dataset does not exists at " + data_path

        df = pd.read_csv(self.data_path, names=[0,1,2,3,4])

        # dic = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}
        df[4] = df[4].map({"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2})

        data = df.iloc[:,:4]
        label = df.iloc[:,4:]

        data = (data - np.mean(data)/np.std(data))

        self.data = torch.from_numpy(np.array(data, dtype='float32'))
        self.label = torch.from_numpy(np.array(label, dtype='int64'))
    
        self.data_num = len(label)

        print("df:", df)
        print("data set size", self.data_num)
        print("data set", self.data)
        print("label set", self.label)

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        return list(self.data)[index], list(self.label)[index]