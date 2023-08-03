import torch
import torch.nn as nn
import torch.nn.functional as F


# Model Definition
class SingleStageTCN(nn.Module):
    def __init__(self, ftrs_embed, embed_dim, temp_window, dropout):
        super(SingleStageTCN, self).__init__()

        # conv layers
        self.convL1 = nn.Conv1d(ftrs_embed,64,temp_window,padding='same')
        self.convL2 = nn.Conv1d(64,32,temp_window,padding='same')
        self.convL3 = nn.Conv1d(32,16,temp_window,padding='same')
        self.convL4 = nn.Conv1d(16,embed_dim,temp_window,padding='same')

        # dropout
        self.dropout = nn.Dropout(dropout)
        # Pooling layer
        self.pool = nn.MaxPool1d(3,1)
        # Norm layers
        self.layernorm1 = nn.BatchNorm1d(64)
        self.layernorm2 = nn.BatchNorm1d(32)
        self.layernorm3 = nn.BatchNorm1d(16)
        self.layernorm4 = nn.BatchNorm1d(embed_dim)

    def forward(self, x):

        # TCN input video stream
        x = self.dropout(self.layernorm1(self.pool(F.relu(self.convL1(x))))) # b,c,L
        x = self.dropout(self.layernorm2(self.pool(F.relu(self.convL2(x))))) 
        x = self.dropout(self.layernorm3(self.pool(F.relu(self.convL3(x))))) 
        out = self.dropout(self.layernorm4(self.pool(F.relu(self.convL4(x))))) 

        return out
    
class force_model(nn.Module):
    def __init__(self, embed_dim):
        super(force_model, self).__init__()

        # TCN
        self.SingleStageTCN = SingleStageTCN(ftrs_embed=1, embed_dim=embed_dim, temp_window=25, dropout=0)

        # Clssification head
        self.head = nn.Linear(embed_dim,1)


    def forward(self, x):
        
        x = self.SingleStageTCN(x)
        
        out = self.head(x.mean(-1))
        
        return out
        
        
        
        
