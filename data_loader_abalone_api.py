from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch

from ucimlrepo import fetch_ucirepo 


class dataloader_abalone_api(Dataset):
    def __init__(self):
       
        # fetch dataset 
        abalone = fetch_ucirepo(id=1) 
  
        # data (as pandas dataframes) 
        X = abalone.data.features 
        y = abalone.data.targets 
  
        # metadata 
        print(abalone.metadata) 
  
        # variable information 
        print(abalone.variables)

        #print(abalone.data.to_string())

        # dic = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}
        # df[4] = df[4].map({"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2})

        # data = df.iloc[:,:4]
        # label = df.iloc[:,4:]

        # data = (data - np.mean(data)/np.std(data))

        # self.data = torch.from_numpy(np.array(data, dtype='float32'))
        # self.label = torch.from_numpy(np.array(label, dtype='int64'))
    
        # self.data_num = len(label)

        # print("df:", df)
        # print("data set size", self.data_num)
        # print("data set", self.data)
        # print("label set", self.label)

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        return list(self.data)[index], list(self.label)[index]