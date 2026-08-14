from torch import nn
class ShakespeareLSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim=64,hidden_size=256,num_layers=2):
        super().__init__(); self.embedding=nn.Embedding(vocab_size,embedding_dim); self.lstm=nn.LSTM(embedding_dim,hidden_size,num_layers=num_layers,batch_first=True); self.head=nn.Linear(hidden_size,vocab_size)
    def forward(self,x,return_features=False):
        embedded=self.embedding(x.long()); out,_=self.lstm(embedded); z=out[:,-1,:]; logits=self.head(z); return (z,logits) if return_features else logits
