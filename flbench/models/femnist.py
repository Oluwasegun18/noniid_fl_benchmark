import torch
from torch import nn

class FEMNISTCNN(nn.Module):
    def __init__(self,num_classes=62):
        super().__init__(); self.features=nn.Sequential(nn.Conv2d(1,32,5,padding=2),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,5,padding=2),nn.ReLU(),nn.MaxPool2d(2)); self.proj=nn.Sequential(nn.Flatten(),nn.Linear(64*7*7,256),nn.ReLU()); self.head=nn.Linear(256,num_classes)
    def forward(self,x,return_features=False):
        z=self.proj(self.features(x)); logits=self.head(z); return (z,logits) if return_features else logits
