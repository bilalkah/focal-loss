from torchvision.transforms.transforms import Compose
import model
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
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
lr = 1e-4
batch_size = 32
epochs = 30

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

# create dataset from directory "cinic10" for torchvision
train_cinic10 = datasets.ImageFolder(root='cinic10/train',
                                     transform=Compose([
                                         transforms.Resize((224, 224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                     ]))

train_loader = DataLoader(
    dataset=train_cinic10,
    batch_size=batch_size,
    shuffle=True,
)


model = model.Darknet19(10).to(device)
criterion1 = CFocalLoss(0.84,2.8).to(device)
optimizer = optim.Adam(model.parameters(),lr=lr)

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

torch.save(model.state_dict(), "./darknet_saved")



