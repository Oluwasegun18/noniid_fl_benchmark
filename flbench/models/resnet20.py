import torch
from torch import nn

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,in_planes,planes,stride=1):
        super().__init__(); self.conv1=nn.Conv2d(in_planes,planes,3,stride=stride,padding=1,bias=False); self.bn1=nn.BatchNorm2d(planes); self.relu=nn.ReLU(inplace=True); self.conv2=nn.Conv2d(planes,planes,3,padding=1,bias=False); self.bn2=nn.BatchNorm2d(planes)
        self.shortcut=nn.Identity() if stride==1 and in_planes==planes else nn.Sequential(nn.Conv2d(in_planes,planes,1,stride=stride,bias=False),nn.BatchNorm2d(planes))
    def forward(self,x):
        out=self.relu(self.bn1(self.conv1(x))); out=self.bn2(self.conv2(out)); return self.relu(out+self.shortcut(x))

class ResNet20(nn.Module):
    def __init__(self,num_classes=100):
        super().__init__(); self.in_planes=16; self.conv=nn.Conv2d(3,16,3,padding=1,bias=False); self.bn=nn.BatchNorm2d(16); self.relu=nn.ReLU(inplace=True); self.layer1=self._make(16,3,1); self.layer2=self._make(32,3,2); self.layer3=self._make(64,3,2); self.pool=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(64,num_classes)
    def _make(self,planes,blocks,stride):
        layers=[BasicBlock(self.in_planes,planes,stride)]; self.in_planes=planes
        for _ in range(1,blocks): layers.append(BasicBlock(self.in_planes,planes,1))
        return nn.Sequential(*layers)
    def forward(self,x,return_features=False):
        x=self.relu(self.bn(self.conv(x))); x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); z=self.pool(x).flatten(1); logits=self.head(z); return (z,logits) if return_features else logits
