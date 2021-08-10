try :
    from . import model
except:
    from model import LeNet
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

class CFocalLoss(nn.Module):
    """
    0 <= alpha <= 1
    0 <= gamma <=5
    """
    def __init__(self,alpha=0.5,gamma=2,epsilon=1e-9,weight=None,size_average=True) -> None:
        super(CFocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self,pred,target):
        pred = F.softmax(pred,dim=1)
        tar = torch.tensor([[0.]*pred.shape[1]]*pred.shape[0],dtype=torch.float,device='cuda' if torch.cuda.is_available() else 'cpu')
        for idx1,idx2 in enumerate(target):
            tar[idx1][int(idx2.item())] = 1.
        pred = pred + self.epsilon
        loss = torch.tensor([0.]*pred.shape[0],dtype=torch.float,device='cuda' if torch.cuda.is_available() else 'cpu')
        for idx in range(target.shape[0]):
            loss[idx] = -torch.mean(
                tar[idx]*self.alpha*torch.pow((1-pred[idx]),self.gamma)*torch.log2(pred[idx]) 
                + 
                (1-tar[idx])*self.alpha*torch.pow((pred[idx]),self.gamma)*torch.log2(1-pred[idx])
            )
        return torch.mean(loss)

class CategoricalCrossentropy(nn.Module):
    def __init__(self,epsilon=1e-9):
        super(CategoricalCrossentropy,self).__init__()
        self.epsilon = epsilon

    def forward(self,pred,target):
        pred = F.softmax(pred,dim=0)
        tar = torch.tensor([[0.]*pred.shape[1]]*pred.shape[0],dtype=torch.float,device='cuda' if torch.cuda.is_available() else 'cpu')
        for idx1,idx2 in enumerate(target):
            tar[idx1][idx2.item()] = 1.
        pred = pred + self.epsilon
        loss = torch.tensor([0.]*pred.shape[0],dtype=torch.float,device='cuda' if torch.cuda.is_available() else 'cpu')
        for idx in range(target.shape[0]):
            loss[idx] = -torch.mean(tar[idx]*torch.log2(pred[idx])+(1-tar[idx])*torch.log2(1-pred[idx]))
        return torch.mean(loss)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
lr = 1e-3
batch_size = 64
epochs = 10

tensor_x = torch.Tensor(np.load("mnist-dataset.npy")) # transform to torch tensor
tensor_y = torch.Tensor(np.load("mnist-target.npy")).type(torch.long)
tensor_x = tensor_x.permute(0,3,1,2)
tensor_x = transforms.functional.resize(tensor_x,(32,32))
my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
print(tensor_x.shape)
print(tensor_y.shape)
train_loader = DataLoader(
    dataset=my_dataset,
    batch_size=batch_size,
    shuffle=True,
    base_seed=42,
) # create your dataloader

model = LeNet().to(device)
alpha = 0.84
gamma = 2.8
print(alpha, gamma)
criterion = nn.CrossEntropyLoss()#CFocalLoss(alpha,gamma)
optimizer = optim.Adam(model.parameters(),lr=lr)
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
    base_seed=42,
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
    base_seed=42,
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