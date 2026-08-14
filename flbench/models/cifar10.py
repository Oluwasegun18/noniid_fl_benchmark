import torch
from torch import nn

class SimpleCIFAR10CNN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.projection=nn.Sequential(nn.Flatten(),nn.Linear(64*8*8,128),nn.ReLU())
        self.classifier=nn.Linear(128,num_classes)
    def forward_features(self,x): return self.projection(self.features(x))
    def forward(self,x,return_features=False):
        z=self.forward_features(x); logits=self.classifier(z)
        return (z,logits) if return_features else logits
