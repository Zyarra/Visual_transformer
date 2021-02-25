import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.nn1 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
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


class LambdaLayer(nn.Module):
    """
    Lambda layer
    """
    def __init__(self, lambd):
        """
        lambd: lambda function to iterate the data over
        """
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
