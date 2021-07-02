from torch._C import dtype
from torchvision.transforms.transforms import Compose
from vgg.model import VGG
import loss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
from time import sleep

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
lr = 1e-3
batch_size = 4
epochs = 1

dataset = datasets.CIFAR10(
    root = 'dataset/',
    train=True,
    transform=Compose([transforms.ToTensor(),transforms.Resize(size=(224,224))]),
    download=True,
)

train_dataset,val_set = data.random_split(dataset,[10000,40000])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)


model = VGG("A",num_classes).to(device)

criterion1 = nn.CrossEntropyLoss()
criterion = loss.FocalLoss(alpha=0.8,gamma=2)
optimizer = optim.Adam(model.parameters(),lr=lr)

for epoch in range(epochs):

    with tqdm.tqdm(train_loader,unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for data, target in tepoch:
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            scores = model(data)
            
            loss = criterion(scores, target)
            
            acc = (target.argmax(-1) == scores.argmax(-1)).cpu().float().detach().numpy()
            train_acc = float(100*acc.sum() / len(acc))
            

            loss.backward()
            optimizer.step()
            
            tepoch.set_postfix(loss = loss.item(),accuracy = train_acc)
            sleep(0.1)

