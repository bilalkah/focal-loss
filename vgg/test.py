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
from torchvision.transforms.transforms import Compose
import losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
lr = 1e-4
batch_size = 16
epochs = 10

# create dataset from directory "cinic10" for torchvision
test_cinic10 = datasets.ImageFolder(root='cinic10/test',
                                     transform=Compose([
                                         transforms.Resize((224, 224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))

test_loader = DataLoader(
    dataset=test_cinic10,
    batch_size=batch_size,
    shuffle=True,
)

model = VGG("A").to(device)
model.load_state_dict(torch.load("./saved_model_1"))
model.eval()

criterion = losses.CFocalLoss(0.8,3)
metric = torchmetrics.Accuracy().to(device)


with tqdm.tqdm(test_loader,unit="batch") as tepoch:
    tepoch.set_description(f"Evaluate for test")
    epoch_loss = []
    epoch_acc = []
    for data, target in tepoch:
            
        data, target = data.to(device), target.to(device)
        scores = model(data)
            
        loss = criterion(scores, target)
        train_acc = metric(scores,target)        
        epoch_loss.append(loss.item())
        epoch_acc.append(train_acc.item())  
            
        tepoch.set_postfix(CFocal = criterion(scores,target).item(),accuracy = train_acc.item())
        sleep(0.1)
    print(f"Epoch {epoch} avg loss: {sum(epoch_loss)/len(epoch_loss)} avg acc: {sum(epoch_acc)/len(epoch_acc)}")
    

test_horse = datasets.ImageFolder(root='cinic10/test/horse',
                                     transform=Compose([
                                         transforms.Resize((224, 224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))

horse_loader = DataLoader(
    dataset=test_horse,
    batch_size=batch_size,
    shuffle=True,
)


with tqdm.tqdm(horse_loader,unit="batch") as tepoch:
    tepoch.set_description(f"Evaluate for horse")
    epoch_loss = []
    epoch_acc = []
    for data, target in tepoch:
            
        data, target = data.to(device), target.to(device)
        scores = model(data)
            
        loss = criterion(scores, target)
        train_acc = metric(scores,target)        
        epoch_loss.append(loss.item())
        epoch_acc.append(train_acc.item())  
            
        tepoch.set_postfix(CFocal = criterion(scores,target).item(),accuracy = train_acc.item())
        sleep(0.1)
    print(f"Epoch {epoch} avg loss: {sum(epoch_loss)/len(epoch_loss)} avg acc: {sum(epoch_acc)/len(epoch_acc)}")