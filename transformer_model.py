import torch
import torch.nn as nn
import torch.nn.functional as F

# Use
# model = transformer(embed_dim=8, dropout=0.2)


# Model Definition
class transformer(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(transformer, self).__init__()
        
        # Single Stage TCN
        self.conv_1x1 = nn.Conv1d(1,embed_dim,1,padding='same')

        # Self-Attention Block
        self.multihead_attn = nn.MultiheadAttention(embed_dim = embed_dim, num_heads =  8, batch_first=True) #, dropout = dropout)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)

        # Fully Connected layer
        self.fc1 = nn.Linear(embed_dim,embed_dim)
        self.fc2 = nn.Linear(embed_dim,1)


    def forward(self, x):

        # x size [b,1,L]
        # 1D conv layer
        x = F.relu(self.conv_1x1(x)) # [b,ftrs,L]
        x = torch.permute(x,(0,2,1)) # [b,L,1]

        # Self-Attention block
        attn_output, _ = self.multihead_attn(x, x, x)

        # Skip Connection
        x = self.bn1(torch.permute(attn_output + x,(0,2,1))) # [b,ftrs,L]
        
        # Temporal Average Pooling and Fully Connected Layer
        x = self.bn2(F.relu(self.fc1(x.mean(dim=-1))) + x.mean(dim=-1))
        
        s = self.fc2(x)

        return s 