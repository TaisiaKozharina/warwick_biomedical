import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.double_conv(x)  



class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.pool_conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        )


    def forward(self, x):
        return  self.pool_conv(x)
    

class BridgeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BridgeBlock, self).__init__()
        self.double_conv = ConvBlock(in_channels, out_channels)


    def forward(self, x):
        return  self.double_conv(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return  self.up_conv(x)


class FinalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return  self.conv(x)

