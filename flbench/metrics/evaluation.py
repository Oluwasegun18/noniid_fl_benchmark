import torch
from torch import nn
from sklearn.metrics import f1_score
@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); crit=nn.CrossEntropyLoss(reduction='sum'); loss=correct=n=0; pred=[]; truth=[]
    for x,y in loader:
        x,y=x.to(device),y.to(device); logits=model(x); loss+=crit(logits,y).item(); p=logits.argmax(1); correct+=(p==y).sum().item(); n+=len(y); pred+=p.cpu().tolist(); truth+=y.cpu().tolist()
    return {'loss':loss/max(n,1),'accuracy':correct/max(n,1),'macro_f1':f1_score(truth,pred,average='macro',zero_division=0)}
