from lenet import arch
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
from model import Model

batch_size = 64
lr = 1e-3
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tensor_x = torch.Tensor(np.load("mnist-dataset/mnist-dataset.npy")) # transform to torch tensor
tensor_y = torch.Tensor(np.load("mnist-dataset/mnist-target.npy")).type(torch.long)
tensor_x = tensor_x.permute(0,3,1,2)
tensor_x = transforms.functional.resize(tensor_x,(32,32))
my_dataset = TensorDataset(tensor_x,tensor_y) # create your dataset

train_loader = DataLoader(
    dataset=my_dataset,
    batch_size=batch_size,
    shuffle=True,
) # create your dataloader

model = arch.LeNet().to(device)
alpha = 0.84
gamma = 2.8

criterion = CFocalLoss(alpha,gamma).to(device)
optimizer = optim.Adam(model.parameters(),lr=lr)
metric = torchmetrics.Accuracy().to(device)

model_ = Model(model,device)

      
zero_x = torch.Tensor(np.load("mnist-dataset/mnist-zero-test.npy")) # transform to torch tensor
zero_y = torch.Tensor(np.load("mnist-dataset/mnist-zero-test-target.npy")).type(torch.long)
zero_x = zero_x.permute(0,3,1,2)
zero_x = transforms.functional.resize(zero_x,(32,32))

my_dataset = TensorDataset(zero_x,zero_y) # create your datset
test_loader = DataLoader(
    dataset=my_dataset,
    batch_size=batch_size,
    shuffle=True,
) # create your dataloader

model_.train(train_loader,test_loader,epochs,batch_size,lr,optimizer,criterion)