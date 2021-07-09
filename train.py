from torch._C import dtype
from torchvision.transforms.transforms import Compose
from vgg.model import VGG
from lenet.model import LeNet
import losses
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
from time import sleep
import torchmetrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
num_classes = 10
lr = 1e-3
batch_size = 128
epochs = 10

dataset = datasets.MNIST(
    root = 'dataset/',
    train=True,
    transform=Compose([transforms.ToTensor(),transforms.Resize(size=(32,32))]),
    download=True,
)

#train_dataset,val_set = data.random_split(dataset,[10000,50000])

train_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
)

test = datasets.MNIST(
    root = 'dataset/',
    train=False,
    transform=Compose([transforms.ToTensor(),transforms.Resize(size=(32,32))]),
    download=True,
)

test_loader = DataLoader(
    dataset=test,
    batch_size=64,
    shuffle=True,
)


model = LeNet().to(device)

criterion2 = losses.CategoricalCrossentropy()
criterion1 = nn.CrossEntropyLoss()
criterion = losses.CFocalLoss(alpha=0.8,gamma=2)
optimizer = optim.Adam(model.parameters(),lr=lr)
metric = torchmetrics.Accuracy().to(device)

for epoch in range(epochs):
    with tqdm.tqdm(train_loader,unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for data, target in tepoch:
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            scores = model(data)
            
            loss = criterion(scores, target)
            
            train_acc = metric(scores,target)        
            

            loss.backward()
            optimizer.step()
            
            tepoch.set_postfix(CFocal = criterion(scores,target).item(),CustomCE = criterion2(scores,target).item(),Official_loss=criterion1(scores,target).item(),accuracy = train_acc.item())
            sleep(0.1)


with tqdm.tqdm(train_loader,unit="batch") as tepoch:
    tepoch.set_description(f"Evaluate")
    for data, target in tepoch:
            
        data, target = data.to(device), target.to(device)

        scores = model(data)
            
        loss = criterion(scores, target)
        
        train_acc = metric(scores,target)        
            
        tepoch.set_postfix(CFocal = criterion(scores,target).item(),CustomCE = criterion2(scores,target).item(),Official_loss=criterion1(scores,target).item(),accuracy = train_acc.item())
        sleep(0.1)


