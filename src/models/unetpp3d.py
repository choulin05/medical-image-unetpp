# src/models/unetpp3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetPlusPlus3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=1, filters=[32,64,128,256,512], deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        Filt = filters
        # encoder
        self.conv0_0 = ConvBlock3D(in_channels, Filt[0])
        self.conv1_0 = ConvBlock3D(Filt[0], Filt[1])
        self.conv2_0 = ConvBlock3D(Filt[1], Filt[2])
        self.conv3_0 = ConvBlock3D(Filt[2], Filt[3])
        self.conv4_0 = ConvBlock3D(Filt[3], Filt[4])
        # decoder blocks (nested)
        self.conv0_1 = ConvBlock3D(Filt[0]+Filt[1], Filt[0])
        self.conv1_1 = ConvBlock3D(Filt[1]+Filt[2], Filt[1])
        self.conv2_1 = ConvBlock3D(Filt[2]+Filt[3], Filt[2])
        self.conv3_1 = ConvBlock3D(Filt[3]+Filt[4], Filt[3])

        self.conv0_2 = ConvBlock3D(Filt[0]*2+Filt[1], Filt[0])
        self.conv1_2 = ConvBlock3D(Filt[1]*2+Filt[2], Filt[1])
        self.conv2_2 = ConvBlock3D(Filt[2]*2+Filt[3], Filt[2])

        self.conv0_3 = ConvBlock3D(Filt[0]*3+Filt[1], Filt[0])
        self.conv1_3 = ConvBlock3D(Filt[1]*3+Filt[2], Filt[1])

        self.conv0_4 = ConvBlock3D(Filt[0]*4+Filt[1], Filt[0])

        # pooling and upsample
        self.pool = nn.MaxPool3d(2)
        self.up = lambda x: F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

        # final conv
        if self.deep_supervision:
            self.final1 = nn.Conv3d(Filt[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv3d(Filt[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv3d(Filt[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv3d(Filt[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv3d(Filt[0], num_classes, kernel_size=1)

    def forward(self, x):
        # enc
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # dec
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            return self.final(x0_4)
