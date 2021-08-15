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
epochs = 30

# create dataset from directory "cinic10" for torchvision
train_cinic10 = datasets.ImageFolder(root='cinic10/train',
                                     transform=Compose([
                                         transforms.Resize((224, 224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                     ]))

train_loader = DataLoader(
    dataset=train_cinic10,
    batch_size=batch_size,
    shuffle=True,
)


model = VGG("A").to(device)
model.load_state_dict(torch.load("./saved_model"))
criterion1 = losses.CFocalLoss(alpha=0.8,gamma=3)
optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)

metric = torchmetrics.Accuracy().to(device)
exit = False
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
            if torch.isnan(loss):
                print(f"Loss returned Nan value!! Break the training process")
                exit = True
                break
            train_acc = metric(scores,target)        
            epoch_loss.append(loss.item())
            epoch_acc.append(train_acc.item())

            loss.backward()
            optimizer.step()
            
            tepoch.set_postfix(CFocal_loss=loss.item(),Accuracy = train_acc.item())
            sleep(0.1)
        print(f"Epoch {epoch} avg loss: {sum(epoch_loss)/len(epoch_loss)} avg acc: {sum(epoch_acc)/len(epoch_acc)}")
        if exit:
            break

torch.save(model.state_dict(), "./saved_model_1")



