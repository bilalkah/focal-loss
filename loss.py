import torch
import torch.nn as nn
import torch.nn.functional as F



"""
https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch?scriptVersionId=51446304&cellId=13
"""

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.8,gamma=2,weight=None,size_average=True) -> None:
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self,inputs,targets):
        
        BCE = F.cross_entropy(inputs,targets,reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        return self.alpha * (1 - BCE_EXP) ** self.gamma * BCE