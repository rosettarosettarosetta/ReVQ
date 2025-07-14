# ------------------------------------------------------------------------------
# ReVQ: Quantize-then-Rectify: Efficient VQ-VAE Training
# Copyright (c) 2025 Borui Zhang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------

import torch.nn as nn

class ResBlockMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlockMLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.act = nn.SiLU(inplace=True)
        self.norm = nn.BatchNorm1d(num_features=out_channels, affine=True)
    
    def forward(self, x):
        residual = x
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        h = self.norm(h)
        h = h + residual
        return h
    
class ResBlockMLPv2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlockMLPv2, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.act = nn.SiLU(inplace=True)
        self.norm = nn.BatchNorm1d(num_features=out_channels, affine=True)
        self.fc3 = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        residual = x
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        h = self.norm(h)
        h = h + self.fc3(residual)
        return h

class MLPDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 num_layers: int):
        super(MLPDecoder, self).__init__()
        self.layers = nn.Sequential(*[
            ResBlockMLP(in_channels, in_channels)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        """
        Args:
            x (tensor): (BS, T, dim)
        """
        BS, C, H, W = x.size()
        x = x.view(BS, -1)
        x = self.layers(x)
        x = self.fc_out(x)
        x = x.view(BS, C, H, W)
        return x
    
class MLPDecoderv2(nn.Module):
    def __init__(self, in_channels: int, 
                 mid_channels: int,
                 out_channels: int,
                 num_layers: int):
        super(MLPDecoderv2, self).__init__()
        self.prelayers = nn.Sequential(*[
            ResBlockMLPv2(in_channels, mid_channels)
            for _ in range(1)
        ])
        self.postlayers = nn.Sequential(*[
            ResBlockMLPv2(mid_channels, mid_channels)
            for _ in range(num_layers - 1)
        ])
        self.fc_out = nn.Linear(mid_channels, out_channels)
    
    def forward(self, x):
        """
        Args:
            x (tensor): (BS, T, dim)
        """
        
        BS, C, H, W = x.size()
        x = x.view(BS, -1)
        x = self.prelayers(x)
        x = self.postlayers(x)
        x = self.fc_out(x)
        x = x.view(BS, C, H, W)
        return x

if __name__ == "__main__":
    pass