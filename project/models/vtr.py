from torch import nn
from project.models.blocks import Transformer
from functools import partial
from project.utils.modulefreezer import freeze
from project.hyperparameters import *


class VTR(nn.Module):
    """
    The last stage of ResNet101 contains 3 bottleneck blocks. At the end of the network, we output 16 visual tokens
    to the classification head, which applies an average pooling over the tokens and use a fully-connected layer to
    predict the probability We replace them with the same number 3 of VT modules. At the end of stage-4 (before
    stage-5 max pooling)

        ResNet-{101} generate 14x14 × 1024.
        We set VT’s feature map channel size 1024
        VT block with a channel size for the output feature map as 1024, channel size for visual tokens as 1024,
        and the number of tokens as 16.
        We adopt 16 visual tokens with a channel size of 1024. Only train the transformer and the FC layers.
    """
    def __init__(self, img_size=IMAGE_SIZE, in_chans=CHANNELS, num_classes=NUM_CLASSES,
                 dim=VTR_DIM, depth=TRANSFORMER_DEPTH,
                 num_heads=TRANSFORMER_HEADS, mlp_hidden_dim=VTR_MLP_DIM,
                 dropout=VTR_DROPOUT, attn_dropout=ATTN_DROPOUT, emb_dropout=EMB_DROPOUT, backbone=BACKBONE):
        """img_size: input image size
            in_chans: number of input channels
            num_classes: number of classes for classification head
            dim :embedding dimension
            depth: depth of transformer
            num_heads : number of attention heads
            mlp_dim : mlp hidden dimension
            drop_rate : dropout rate
            attn_drop_rate : attention dropout rate
            drop_path_rate : stochastic depth rate
            backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
        """
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.img_size = (img_size, img_size)
        self.in_chans = in_chans
        backbone = nn.Sequential(*list(backbone.children())[:-3])
        freeze(module=backbone, train_bn=False)
        self.backbone = backbone
        feature_map = backbone(torch.zeros(1, in_chans, img_size, img_size))
        feature_size = feature_map.shape[-2:]
        feature_dim = feature_map.shape[1]
        num_patches = feature_size[0] * feature_size[1]
        self.patch_embed = nn.Conv2d(feature_dim, dim, 1).cuda()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim=dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim,
                                       dropout=dropout, attn_dropout=attn_dropout, depth=depth)
        self.norm = norm_layer(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x = self.emb_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)[:, 0]
        return self.fc(x)
