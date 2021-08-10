import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalCrossentropy(nn.Module):
    def __init__(self,epsilon=1e-9):
        super(CategoricalCrossentropy,self).__init__()
        self.epsilon = epsilon

    def forward(self,pred,target):
        pred = F.softmax(pred,dim=0)
        tar = torch.tensor([[0.]*pred.shape[1]]*pred.shape[0],dtype=torch.float,device='cuda' if torch.cuda.is_available() else 'cpu')
        for idx1,idx2 in enumerate(target):
            tar[idx1][idx2.item()] = 1.
        pred = pred + self.epsilon
        loss = torch.tensor([0.]*pred.shape[0],dtype=torch.float,device='cuda' if torch.cuda.is_available() else 'cpu')
        for idx in range(target.shape[0]):
            loss[idx] = -torch.mean(tar[idx]*torch.log2(pred[idx])+(1-tar[idx])*torch.log2(1-pred[idx]))
        return torch.mean(loss)

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

class BinaryCrossentropy(nn.Module):
    def __init__(self):
        super(BinaryCrossentropy,self).__init__()

    def forward(self,pred,target):
        pred = pred.view(2)
        return -torch.mean(target*torch.log2(pred[0])+(1-target)*torch.log2(1-pred[0]))



"""
https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch?scriptVersionId=51446304&cellId=13
"""
class BFocalLoss(nn.Module):
    
    """
    https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch?scriptVersionId=51446304&cellId=13
    0 <= alpha <= 1
    0 <= gamma <=5
    """
    
    def __init__(self, weight=None, size_average=True):
        super(BFocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss