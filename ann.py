import torch
import torch.nn as nn
import losses

class ANN(nn.Module):
    def __init__(self):
        super(ANN,self).__init__()
        self.fc1 = nn.Linear(10,6)
        self.fc2 = nn.Linear(6,9)
        self.fc3 = nn.Linear(9,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)

model = ANN().to('cuda')
x = torch.randn((2,10),dtype=torch.float32,device='cuda')
y = torch.tensor([0,1],dtype=torch.float32,device='cuda')
print(model(x).shape)
loss = losses.BinaryCrossentropy()
loss(model(x),y)

