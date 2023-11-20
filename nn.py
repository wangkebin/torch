import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
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
ttl = DataLoader(tts, batch_size=1, shuffle=False)

# helper

def infer(model, dataset, device):
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for data in dataset:
            datas, label = data
            outputs = model(datas.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc_num += torch.eq(predict_y, label.to(device)).sum().item()
    acc = acc_num / len(dataset)
    return acc

def main(lr=0.005, epochs = 20):
    model = neuNet(4, 12, 6, 3)
    loss_f = nn.CrossEntropyLoss()

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=lr)

    #weight 
    save_path = os.path.join(os.getcwd(), "results/weights")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    #training
    for epoch in range(epochs):
        model.train()
        acc_num = torch.zeros(1).to(device)
        sample_num = 0

        train_bar = tqdm(tl, file=sys.stdout, ncols=100)
        for datas in train_bar:
            data, label = datas
            label = label.squeeze(-1)
            sample_num += data.shape[0]

            optimizer.zero_grad()
            outputs = model(data.to(device))
            pred_class = torch.max(outputs, dim=1)[1] #torch.mx returns a collection, first is max, second is its index
            acc_num = torch.eq(pred_class, label.to(device)).sum()

            loss = loss_f(outputs, label.to(device))
            loss.backward()
            optimizer.step()

            train_acc = acc_num / sample_num
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f} train_acc:{:.3f}".format(epoch+1, epochs, loss, train_acc)

        val_acc = infer(model, vl, device)
        print("train epoch [{}/{}] loss:{:.3f} train_acc:{:.3f} val_acc:{:.3f}".format(epoch+1, epochs, loss, train_acc, val_acc))
        torch.save(model.state_dict(), os.path.join(save_path, "nn.pth"))

        #data cleanup
        train_acc = 0.
        val_acc = 0.
    print("Finished training")
    test_acc = infer(model, ttl, device)
    print("test_acc:", test_acc)

if __name__ == "__main__":
    main(lr=0.005)




