from torch import nn
import torch
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from utils.helpers import *
from models.blocks import Transformer
from functools import partial
from utils.modulefreezer import freeze


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
    def __init__(self, img_size, in_chans, num_classes,
                 dim, depth,
                 num_heads, mlp_hidden_dim,
                 dropout, attn_dropout, emb_dropout, backbone, backbone_repo):
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
        backbone = torch.hub.load(backbone_repo, backbone, pretrained=True, verbose=False)
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


class TiT(nn.Module):
    def __init__(self, *, image_size, patch_dim, pixel_dim, patch_size, pixel_size, depth, num_classes, inner_heads=4,
                 outer_heads=6, dim_head_inner=24, dim_head_outer=64, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        assert divisible_by(image_size, patch_size), 'image size must be divisible by patch size'
        assert divisible_by(patch_size, pixel_size), 'patch size must be divisible by pixel size for now'

        num_patch_tokens = (image_size // patch_size) ** 2
        pixel_width = patch_size // pixel_size
        num_pixels = pixel_width ** 2
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_tokens = nn.Parameter(torch.randn(num_patch_tokens + 1, patch_dim))
        self.to_pixel_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1=patch_size, p2=patch_size),
            nn.Unfold(pixel_size, stride=pixel_size),
            Rearrange('... c n -> ... n c'),
            nn.Linear(3 * pixel_size ** 2, pixel_dim)
        )

        self.patch_pos_emb = nn.Parameter(torch.randn(num_patch_tokens + 1, patch_dim))
        self.pixel_pos_emb = nn.Parameter(torch.randn(num_pixels, pixel_dim))

        layers = nn.ModuleList([])
        for _ in range(depth):
            pixel_to_patch = nn.Sequential(
                nn.LayerNorm(pixel_dim),
                Rearrange('... n d -> ... (n d)'),
                nn.Linear(pixel_dim * num_pixels, patch_dim),
            )

            layers.append(nn.ModuleList([
                LayerNormalize(pixel_dim, Attention(dim=pixel_dim, heads=inner_heads, dim_head=dim_head_inner, dropout=attn_dropout)),
                LayerNormalize(pixel_dim, FeedForward(dim=pixel_dim, dropout=dropout)),
                pixel_to_patch,
                LayerNormalize(patch_dim, Attention(dim=patch_dim, heads=outer_heads, dim_head=dim_head_outer, dropout=attn_dropout)),
                LayerNormalize(patch_dim, FeedForward(dim=patch_dim, dropout=dropout)),
            ]))

        self.layers = layers

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, num_classes)
        )

    def forward(self, x):
        b, _, h, w, patch_size, image_size = *x.shape, self.patch_size, self.image_size
        assert divisible_by(h, patch_size) and divisible_by(w,
                                                            patch_size), f'height {h} and width {w} of input must be divisible by the patch size'

        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        n = num_patches_w * num_patches_h

        pixels = self.to_pixel_tokens(x)
        patches = repeat(self.patch_tokens[:(n + 1)], 'n d -> b n d', b=b)

        patches += rearrange(self.patch_pos_emb[:(n + 1)], 'n d -> () n d')
        pixels += rearrange(self.pixel_pos_emb, 'n d -> () n d')

        for pixel_attn, pixel_ff, pixel_to_patch_residual, patch_attn, patch_ff in self.layers:
            pixels = pixel_attn(pixels) + pixels
            pixels = pixel_ff(pixels) + pixels

            patches_residual = pixel_to_patch_residual(pixels)

            patches_residual = rearrange(patches_residual, '(b h w) d -> b (h w) d', h=num_patches_h, w=num_patches_w)
            patches_residual = F.pad(patches_residual, (0, 0, 1, 0), value=0)  # cls token gets residual of 0
            patches = patches + patches_residual

            patches = patch_attn(patches) + patches
            patches = patch_ff(patches) + patches

        cls_token = patches[:, 0]
        return self.mlp_head(cls_token)
