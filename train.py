from torchvision.transforms.transforms import Compose
from vgg.model import VGG
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
import losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
lr = 1e-4
batch_size = 32
epochs = 10

# create dataset from directory "cinic10" for torchvision
train_cinic10 = datasets.ImageFolder(root='cinic10/train',
                                     transform=Compose([
                                         transforms.Resize((224, 224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))

train_loader = DataLoader(
    dataset=train_cinic10,
    batch_size=batch_size,
    shuffle=True,
)


model = VGG("A").to(device)

criterion1 = losses.CFocalLoss(alpha=0.8,gamma=3)
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
            
            loss = criterion1(scores, target)
            train_acc = metric(scores,target)        
            epoch_loss.append(loss.item())
            epoch_acc.append(train_acc.item())

            loss.backward()
            optimizer.step()
            
            tepoch.set_postfix(Official_loss=loss.item(),accuracy = train_acc.item())
            sleep(0.1)
        print(f"Epoch {epoch} avg loss: {sum(epoch_loss)/len(epoch_loss)} avg acc: {sum(epoch_acc)/len(epoch_acc)}")

torch.save(model.state_dict(), "./saved_model")



