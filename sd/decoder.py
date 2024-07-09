import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (Batch_Size, Channels, Height, Width)

        residue = x

        n, c, h, w = x.shape # batch_size, channels, height, width

        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height * Width)
        x = x.view(n,c,h*w)

        # (Batch_Size, Channels, Height * Width) -> (Batch_Size, Height * Width, Channels)
        x = x.transpose(-1, -2)

        # (Batch_Size, Height * Width, Channels) -> (Batch_Size, Height * Width, Channels)
        x = self.attention(x)

        # (Batch_Size, Height * Width, Channels) -> (Batch_Size, Channels, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, Channels, Height * Width) -> (Batch_Size, Channels, Height, Width)
        x = x.view((n, c, h, w))

        x += residue

        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels) # group normalization
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # convolution

        self.groupnorm_2 = nn.GroupNorm(32, out_channels) # group normalization
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # convolution

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (Batch_Size, in_channels, Height, Width)

        residue = x # residual

        x = self.groupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residue)
    
