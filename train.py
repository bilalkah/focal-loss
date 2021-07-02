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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
lr = 1e-3
batch_size = 4
epochs = 1

train_dataset = datasets.CIFAR10(
    root = 'dataset/',
    train=True,
    transform=Compose([transforms.ToTensor(),transforms.Resize(size=(224,224))]),
    download=True,
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
test_dataset = datasets.CIFAR10(
    root='dataset/',
    train=False,
    transform=Compose([transforms.ToTensor(),transforms.Resize(size=(224,224))]),
    download=True,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
)

model = VGG("A",num_classes).to(device)

criterion1 = nn.CrossEntropyLoss()
criterion = loss.FocalLoss(alpha=0.8,gamma=2)
optimizer = optim.Adam(model.parameters(),lr=lr)

for epoch in range(epochs):

    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores,targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        print(f"{batch_idx}, loss:{loss}")

    print(f"epoch {epoch}")
