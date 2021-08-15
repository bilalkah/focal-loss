from lenet import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import TensorDataset,DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import tqdm
from time import sleep
import torchmetrics
import torch.nn.functional as F
from losses import CFocalLoss
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
lr = 1e-3
batch_size = 64
epochs = 10

tensor_x = torch.Tensor(np.load("mnist-dataset/mnist-dataset.npy")) # transform to torch tensor
tensor_y = torch.Tensor(np.load("mnist-dataset/mnist-target.npy")).type(torch.long)
tensor_x = tensor_x.permute(0,3,1,2)
tensor_x = transforms.functional.resize(tensor_x,(32,32))
my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
print(tensor_x.shape)
print(tensor_y.shape)
train_loader = DataLoader(
    dataset=my_dataset,
    batch_size=batch_size,
    shuffle=True,
) # create your dataloader

model = model.LeNet().to(device)
alpha = 0.84
gamma = 2.8
print(alpha, gamma)
criterion = CFocalLoss(alpha,gamma).to(device)
optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
metric = torchmetrics.Accuracy().to(device)

for epoch in range(epochs):
    with tqdm.tqdm(train_loader,unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        epoch_loss = []
        epoch_acc = []
        for data, target in tepoch:
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            scores = model(data)
            
            loss = criterion(scores, target)

            train_acc = metric(scores,target)        
            epoch_loss.append(loss.item())
            epoch_acc.append(train_acc.item())
            
            loss.backward()
            optimizer.step()
            
            tepoch.set_postfix(FocalLoss=loss.item(),accuracy = train_acc.item())
            sleep(0.1)
        print(f"Epoch {epoch} avg loss: {sum(epoch_loss)/len(epoch_loss)} avg acc: {sum(epoch_acc)/len(epoch_acc)}")
      
zero_x = torch.Tensor(np.load("mnist-zero-dataset.npy")) # transform to torch tensor
zero_y = torch.Tensor(np.load("mnist-zero-target.npy")).type(torch.long)
zero_x = zero_x.permute(0,3,1,2)
zero_x = transforms.functional.resize(zero_x,(32,32))
print(zero_x.shape)
print(zero_y.shape)
my_dataset = TensorDataset(zero_x,zero_y) # create your datset
train_loader = DataLoader(
    dataset=my_dataset,
    batch_size=batch_size,
    shuffle=True,
) # create your dataloader

with tqdm.tqdm(train_loader,unit="batch") as tepoch:
    tepoch.set_description(f"Evaluate train data")
    for data, target in tepoch:
            
        data, target = data.to(device), target.to(device)

        scores = model(data)
        loss = criterion(scores, target)
        
        train_acc = metric(scores,target)        
            
        tepoch.set_postfix(FocalLoss=loss.item(),accuracy = train_acc.item())
        sleep(0.1)
      
zero_x = torch.Tensor(np.load("mnist-zero-test.npy")) # transform to torch tensor
zero_y = torch.Tensor(np.load("mnist-zero-test-target.npy")).type(torch.long)
zero_x = zero_x.permute(0,3,1,2)
zero_x = transforms.functional.resize(zero_x,(32,32))
print(zero_x.shape)
print(zero_y.shape)
my_dataset = TensorDataset(zero_x,zero_y) # create your datset
train_loader = DataLoader(
    dataset=my_dataset,
    batch_size=batch_size,
    shuffle=True,
) # create your dataloader

with tqdm.tqdm(train_loader,unit="batch") as tepoch:
    tepoch.set_description(f"Evaluate test data")
    epoch_loss = []
    epoch_acc = []
    for data, target in tepoch:
            
        data, target = data.to(device), target.to(device)

        scores = model(data)
        loss = criterion(scores, target)
        
        train_acc = metric(scores,target)        
        epoch_loss.append(loss.item())
        epoch_acc.append(train_acc.item())   
        
        tepoch.set_postfix(FocalLoss=loss.item(),accuracy = train_acc.item())
        sleep(0.1)
    print(f"Epoch {epoch} avg loss: {sum(epoch_loss)/len(epoch_loss)} avg acc: {sum(epoch_acc)/len(epoch_acc)}")