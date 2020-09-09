import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms


import matplotlib.pyplot as plt
import time


cuda = torch.cuda.is_available()

c_size = 17
device = torch.device("cuda" if cuda else "cpu")

class MyDataset_test(Dataset):
    def __init__(self, X, Y):     
        self.X = np.vstack(X)
        self.Y = np.hstack(Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self,index):
        if index >= c_size:
            if index +c_size < self.X.shape[0]:
                X = self.X[index-c_size:index+c_size +1].reshape(-1)
                m = nn.ConstantPad1d((0, 0), 0.)
                X = m(torch.Tensor(X)).float()
            else:
                m = nn.ConstantPad1d((0, 40*(index+c_size +1 - self.X.shape[0])), 0.)
                X = self.X[index-c_size:self.X.shape[0]].reshape(-1)
                X = m(torch.Tensor(X)).float()         
        else:
            m = nn.ConstantPad1d((40*(c_size-index), 0), 0.)
            X = self.X[0:index+c_size+1].reshape(-1)
            X = m(torch.Tensor(X)).float()
            
        Y = self.Y[index]
        return X,Y


class MyDataset_train(Dataset):
    def __init__(self, X, Y,transform=None):
        self.transform = transform
        for i in range(len(X)):
            X[i] = np.pad(X[i], ((5,5), (0,0)), 'constant',constant_values=(0.0))
            Y[i] = np.pad(Y[i],(5,5),'constant', constant_values=(138))

        self.X = np.vstack(X)
        self.Y = np.hstack(Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self,index):
        if index < 5:
            X = self.X[0:index+6]
            X = np.pad(X, ((5-index,0), (0,0)), 'constant').reshape(-1)
        elif len(self.Y) - index <= 5:
            X = self.X[index-5:len(self.Y)]
            X = np.pad(X, ((0,5 -(len(self.Y)-1-index)), (0,0)), 'constant').reshape(-1)
        else:
            X = self.X[index-5:index+6].reshape(-1)
            
        Y = self.Y[index]
        return torch.Tensor(X).float(),Y

# SIMPLE MODEL DEFINITION
class Simple_MLP(nn.Module):
    def __init__(self, size_list):
        super(Simple_MLP, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            #layers.append(nn.Dropout(0.2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size_list[i+1]))
            
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_epoch(model, train_loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):   
        optimizer.zero_grad()   # .backward() accumulates gradients
        data = data.to(device)
        target = target.to(device) # all data & model on same device
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    end_time = time.time()
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss


def test_model(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(test_loader):   
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()
            

        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc

def test_model_kaggle(model, testK_loader, criterion):
    with torch.no_grad():
        model.eval()
        result ={'id':[],'label':[]}
        for batch_idx, (data, target) in enumerate(testK_loader):   
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            result['id'].append(batch_idx)
            result['label'].append(predicted.cpu().numpy()[0])

            if batch_idx % 10000 == 0:
                print(batch_idx)   
        return result

def learning_rate_decay(optim,epoch,lr):
    lr = lr * (0.1 ** (epoch//30))
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def main():
    from sklearn.utils import shuffle

    num_workers = 0 if cuda else 0 
   
    model = Simple_MLP([(c_size * 2 + 1) * 40,2048, 1024,512,256,512,1024,139])
    criterion = nn.CrossEntropyLoss()
    model.cuda()
    print(device)
    print(model)


    load_model = True
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    

    #load model
    if load_model:
        checkpoint = torch.load('./checkpoints/0219_v4model_epoch_48.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
         start_epoch = 0
    print("Resume training at epoch: ", start_epoch)

   
        
    testK_x = np.load('./data/test.npy')
    # Testing
    testK_dataset = MyDataset_test(testK_x, np.zeros((223592)))
    print("debug: ", testK_dataset.X.shape)
    testK_loader_args = dict(shuffle=False, batch_size=1, num_workers=num_workers, pin_memory=True) if cuda\
                        else dict(shuffle=False, batch_size=1)
    testK_loader = DataLoader(testK_dataset, **testK_loader_args)

    r = test_model_kaggle(model, testK_loader, criterion)
    import pandas as pd
    df = pd.DataFrame(r)

    df.to_csv('./data/result0212_v2.csv', index=False)

if __name__ == '__main__':
    main()
