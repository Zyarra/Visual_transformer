import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_dropout=0.1, dropout=0.1, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        hidden_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, hidden_dim*3, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.nn1 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        q, k, v = self.qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.nn1(x)
        x = self.dropout(x)
        return x


class Residual(nn.Module):
    """
    In a network with residual blocks, each layer feeds into the next layer and directly
    into the layers about 2–3 hops away
    """
    def __init__(self, fn):
        """
        fn: layer function to apply the residual layer on
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    """Applies Layer Normalization over a mini-batch of inputs"""
    def __init__(self, dim, fn):
        """
        dim: input/output dimensions
        fn: layer function to apply the normalization on
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

