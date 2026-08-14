from torch import nn
class TabularMLP(nn.Module):
    def __init__(self,input_dim,num_classes=7,hidden=(128,64)):
        super().__init__(); layers=[]; d=input_dim
        for h in hidden: layers += [nn.Linear(d,h),nn.ReLU()]; d=h
        self.features=nn.Sequential(*layers); self.head=nn.Linear(d,num_classes)
    def forward(self,x,return_features=False):
        z=self.features(x.float()); logits=self.head(z); return (z,logits) if return_features else logits
