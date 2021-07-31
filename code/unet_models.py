#!/usr/bin/env python
# coding: utf-8

# In[33]:

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet_encoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.max_pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(82944, 4096)
        self.fc2 = nn.Linear(4096, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.max_pool(x5)
        x5 = x5.reshape([x5.shape[0], -1])
        x_fc = self.fc1(x5)
        out = self.fc2(x_fc)
        return out

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet_small(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_small, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        mult = 2
        
        self.inc = DoubleConv(n_channels, 2*mult)
        self.down1 = Down(2*mult, 4*mult)
        self.down2 = Down(4*mult, 8*mult)
        self.down3 = Down(8*mult, 16*mult)
        factor = 2 if bilinear else 1
        self.down4 = Down(16*mult, 32*mult // factor)
        self.up1 = Up(32*mult, 16*mult // factor, bilinear)
        self.up2 = Up(16*mult, 8*mult // factor, bilinear)
        self.up3 = Up(8*mult, 4*mult // factor, bilinear)
        self.up4 = Up(4*mult, 2*mult, bilinear)
        self.outc = OutConv(2*mult, n_classes)
        
        self.down5 = Down(2, 4*mult)
        self.down6 = Down(4*mult, 8*mult)
        self.down7 = Down(8*mult, 16*mult)
        self.down8 = Down(16*mult, 8*mult)
        self.down9 = Down(8*mult, mult)
        
        self.fc1_1 = nn.Linear(306, 512)
        self.fc1_2 = nn.Linear(512, 128)
        self.fc1_3 = nn.Linear(128, 315 * 50)
        
        self.fc2_1 = nn.Linear(306, 512)
        self.fc2_2 = nn.Linear(512, 128)
        self.fc2_3 = nn.Linear(128, 560 * 50)
        
        self.softmax = nn.Softmax()

        

    def forward(self, x_in):
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        
        
        
        
        combined = torch.cat((logits, x_in), axis=1)
        x6 = self.down5(combined)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)
        x10 = self.down9(x9)
        
        x11 = torch.reshape(x10, (x10.size()[0], -1))
        
        #torch.Size([8, 315, 1])
        #torch.Size([8, 1, 560])
        
        x1_1 = F.tanh(self.fc1_1(x11))
        x1_2 = F.tanh(self.fc1_2(x1_1))
        x1_3 = F.tanh(self.fc1_3(x1_2))
        
        #x1_3 = torch.unsqueeze(x1_3, 2)
        x1_3 = torch.reshape(x1_3, (8, 315, 50))

        x2_1 = F.tanh(self.fc2_1(x11))
        x2_2 = F.tanh(self.fc2_2(x2_1))
        x2_3 = F.tanh(self.fc2_3(x2_2))

        #x2_3 = torch.unsqueeze(x2_3, 1)
        x2_3 = torch.reshape(x2_3, (8, 50, 560))

        
        X_big = torch.matmul(x1_3, x2_3)
        X_big = torch.unsqueeze(X_big, 1)
        
        
        return logits, X_big
