import torch.nn as nn
from models.layers import LayerNormalize, Residual, Attention


class MLP_Block(nn.Module):
    def __init__(self, in_features, hidden_features=None, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Transformer(nn.Module):
    """
    dim: the input and output dimension of the transformer model
    heads: the number of heads in the multi-head-attention models.
    depth: the number of encoder and decoder layers
    mlp_dim: the dimension of the feedforward network model
    dropout: the dropout value.
    """

    def __init__(self, dim, num_heads, depth, mlp_hidden_dim, dropout=01., attn_dropout=01.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(
                    dim, num_heads=num_heads, attn_dropout=attn_dropout, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(
                    in_features=dim, hidden_features=mlp_hidden_dim, dropout=dropout)))
            ]))

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x)

            x = mlp(x)
        return x
