import torch
import torch.nn as nn
import torch.functional as F

try:
    from . import config
except ImportError:
    print("Ralative import failed")

class FC(nn.Module):
    def __init__(self,in_features,out_features):
        super(FC,self).__init__()
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features
        )
        self.relu = nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.fc(x))

class VGG(nn.Module):
    def __init__(self,version,classNum):
        super(VGG,self).__init__()
        self._version = version
        self.arch = config.ConvNet_Config[self._version]
        self.features = self.buildFeatures(self.arch)
        self.fc = self.buildFC(classNum)

    def buildFeatures(self,arch):
        in_channels = 3
        features = []
        for _ ,block in enumerate(arch):
            for _ ,layer in enumerate(block):
                features.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=layer[1],
                    kernel_size=layer[0],
                    stride=1,
                    padding=1
                ))
                features.append(nn.ReLU())
                in_channels = layer[1]
            features.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*features)

    def buildFC(self,classNum):
        in_features = 512*7*7 if self._version != "C" else 512*8*8
        fc = []
        for _ , layer in enumerate(config.FullyConnected):
            fc.append(FC(in_features,layer))
            in_features = layer
        fc.append(FC(in_features,classNum))
        return nn.Sequential(*fc)

    def forward(self,x):
        x = self.features(x).reshape(x.shape[0],-1)
        return nn.Softmax()(self.fc(x))


def test():
    model = VGG("A",10)
    x = torch.randn(4,3,224,224)
    print(model(x).shape)

if __name__ == "__main__":
    test()