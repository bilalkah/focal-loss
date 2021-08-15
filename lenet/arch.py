import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=6,
            kernel_size=(5,5),
        )
        self.avgpool = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5,5),
        )
        self.avgpool1 = nn.AvgPool2d(2,2)
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5,5),
        )
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,num_classes)
        

    def forward(self,x):
        x = torch.tanh(self.conv1(x))
        x = self.avgpool(x)
        x = torch.tanh(self.conv2(x))
        x = self.avgpool1(x)
        x = torch.tanh(self.conv3(x))
        x = x.reshape(x.shape[0],-1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        #x = F.softmax(x,dim=0)
        return x

