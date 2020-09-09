#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch import nn
from torch.nn.utils.rnn import *

from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time

import os
#from ctcdecode import CTCBeamDecoder
from phoneme_list import PHONEME_MAP
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")



class MyDataset(Dataset):
    def __init__(self, X, Y=None,transform=None):
        self.x = X
        self.y = Y
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,index):
        #print("test: ", self.y[index])
        if self.y != None:
            return torch.Tensor(self.x[index]).to(device), torch.Tensor(self.y[index] + 1).to(device) #torch.Tensor(Y).float()
        else:
            return torch.Tensor(self.x[index]).to(device), torch.Tensor([0]).to(device)



class Model(nn.Module):
    def __init__(self, out_vocab, embed_size, hidden_size, num_layers):
        super(Model, self).__init__()
        #self.embed = nn.Embedding(in_vocab, embed_size)
        self.cnn1 = nn.Conv1d(embed_size, 256, kernel_size=3, stride=1)
        self.cnn2 = nn.Conv1d(256, 256, kernel_size=3, stride=1)

        self.lstm = nn.LSTM(256, hidden_size, num_layers, bidirectional=True)
        self.linear1 = nn.Linear(hidden_size * 2, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, out_vocab)
    
    def forward(self, X):
        
        X = self.cnn1(X)
        X = self.cnn2(X)

        X_lens = torch.LongTensor([len(seq) for seq in X])
        X = pad_sequence(X)
        packed_X = pack_padded_sequence(X, X_lens, enforce_sorted=False)
        
        packed_out = self.lstm(packed_X)[0]
        out, out_lens = pad_packed_sequence(packed_out)
        #Log softmax after output layer is required since`nn.CTCLoss` expects log probabilities.
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.output(out).log_softmax(2)
        return out, out_lens
    
# collate fn lets you control the return value of each batch
# for packed_seqs, you want to return your data sorted by length
def collate_lines(seq_list):
    
    #print("debug:", type(seq_list))
    inputs,targets = zip(*seq_list)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets = [targets[i] for i in seq_order]
    return inputs,targets



# In[6]:


#print(hw3_dataset[0][0].shape,hw3_dataset[1][0].shape)
#hw3_dataset[0][1].shape


# In[7]:



def train_epoch(model, train_loader, criterion, optimizer):

    model.train()
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):   
        optimizer.zero_grad()
        #print( len(data),len(target),target)#data.shape, target.shape)
        #data = data.to(device)
        #target = target.to(device) 
        Y_lens = torch.LongTensor([len(seq) for seq in target])
        target = pad_sequence(target, batch_first=True)
        out, out_lens = model(data)
        #print(out.shape, target.shape, out_lens.shape, Y_lens.shape)
        loss = criterion(out, target, out_lens, Y_lens)
        running_loss+=loss.item()
        loss.backward()
        optimizer.step()
    
    end_time = time.time() 
    running_loss /= len(train_loader)
    print('Epoch', epoch + 1,'Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    
    

def test_epoch(model, val_loader, criterion, optimizer):

    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(val_loader): 
            Y_lens = torch.LongTensor([len(seq) for seq in target])
            target = pad_sequence(target, batch_first=True)
            out, out_lens = model(data)
            loss = criterion(out, target, out_lens, Y_lens)

            running_loss+=loss.item()
        running_loss /= len(val_loader)
        print('Testing Loss: ', running_loss)

def inference(model, val_loader):
    import pandas as pd
    result = []
    with torch.no_grad():
        model.eval()
        
        for batch_idx, (data, target) in enumerate(val_loader): 
            Y_lens = torch.LongTensor([len(seq) for seq in target])
            target = pad_sequence(target, batch_first=True)
            out, out_lens = model(data)
            #print("Debug: ", out.shape,out_lens.shape)
            
            decoder = CTCBeamDecoder([' '] +PHONEME_MAP, beam_width=40, num_processes=os.cpu_count(),log_probs_input=True)
            out = out.permute(1, 0, 2)
            decoded, _, _, out_lens = decoder.decode(out, out_lens)
            #result.append()
            #print(decoded[0, 0, :out_lens[0, 0]])
            if out_lens[0, 0] != 0:
                mapped = "".join(PHONEME_MAP[int(s)-1] for s in decoded[0, 0, :out_lens[0, 0]])
                result.append([batch_idx,mapped])
                #print(mapped)
            if batch_idx % 100 == 0:
                print(batch_idx , "...")


    df = pd.DataFrame(np.array(result),columns = ['id','Predicted'])
    print(df)
    df.to_csv('./result.csv',index=False)

    

def main():
    cuda = torch.cuda.is_available()
    train = True

    num_workers = 0 if cuda else 0 
    
    if train:
        print("Loading Data!")
        print('='*20)
        x_train = np.load('./wsj0_train.npy')
        y_train = np.load('./wsj0_train_merged_labels.npy')

        x_dev = np.load('./wsj0_dev.npy')
        y_dev = np.load('./wsj0_dev_merged_labels.npy')
        #x, y = shuffle(x, y)


        print("Loading Ends!")
        print('='*20)

        train_dataset = MyDataset(x_train,y_train)
        val_dataset = MyDataset(x_dev,y_dev)

        train_loader_args = dict(shuffle=True, batch_size=32, num_workers=num_workers, collate_fn = collate_lines) if cuda                    else dict(shuffle=True, batch_size=64)
        val_loader_args = dict(shuffle=False, batch_size=32, num_workers=num_workers, collate_fn = collate_lines) if cuda                    else dict(shuffle=True, batch_size=64)
        train_loader = DataLoader(train_dataset, **train_loader_args)
        val_loader = DataLoader(val_dataset, **train_loader_args)

        model = Model(47, 40, 256,3)
        model = model.to(device)
        criterion = nn.CTCLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
        n_epochs = 20

        for epoch in range(n_epochs):
            train_epoch(model, train_loader, criterion, optimizer)
            test_epoch(model, train_loader, criterion, optimizer)
            print('='*20)

            torch.save({
            'epoch': n_epochs + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }, '%s/0505 model_epoch_%d.pth' % ('./checkpoints', epoch))

    else:
        #inference
        

        print("Inference!")
        print("Loading Data!")
        x_pred = np.load('./wsj0_test.npy', allow_pickle=True)
        print("Loading Ends!")
        print('='*20)

        test_dataset = MyDataset(x_pred)
        test_loader_args = dict(shuffle=False, batch_size=1, num_workers=num_workers, collate_fn = collate_lines) if cuda else dict(shuffle=False, batch_size=1)
        test_loader = DataLoader(test_dataset, **test_loader_args)

        model = Model(47, 40, 256,3)
        checkpoint = torch.load('./test.pth')

        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)


        inference(model, test_loader)


        



if __name__ == '__main__':
    main()