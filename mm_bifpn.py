import sys
sys.path.append("../..")

from utils import *
from backbone import *
from bifpn import BiFPNUnit

import torch
import torch.nn as nn

def croppCenter(tensorToCrop, finalShape):
    org_shape = tensorToCrop.shape

    diff = np.zeros(2)
    diff[0] = org_shape[2] - finalShape[2]
    diff[1] = org_shape[3] - finalShape[3]

    croppBorders = np.zeros(2, dtype=int)
    croppBorders[0] = int(diff[0] / 2)
    croppBorders[1] = int(diff[1] / 2)

    return tensorToCrop[:, :, croppBorders[0]:croppBorders[0] + finalShape[2],
           croppBorders[1]:croppBorders[1] + finalShape[3]]

class MM_BiFPN(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=32):
        super(MM_BiFPN, self).__init__()
        print('~' * 50)
        print(' ---- Creating Multi UNet ---')
        print('~' * 50)

        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc

        # Encoder (Modality 1) Flair 1
        self.down_1_0 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_0 = maxpool()
        self.down_2_0 = ConvBlock2d(self.out_dim, self.out_dim * 2)
        self.pool_2_0 = maxpool()
        self.down_3_0 = ConvBlock2d(self.out_dim * 2, self.out_dim * 4)
        self.pool_3_0 = maxpool()
        self.down_4_0 = ConvBlock2d(self.out_dim * 4, self.out_dim * 8)
        self.pool_4_0 = maxpool()

        # Encoder (Modality 2) T1
        self.down_1_1 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_1 = maxpool()
        self.down_2_1 = ConvBlock2d(self.out_dim, self.out_dim * 2)
        self.pool_2_1 = maxpool()
        self.down_3_1 = ConvBlock2d(self.out_dim * 2, self.out_dim * 4)
        self.pool_3_1 = maxpool()
        self.down_4_1 = ConvBlock2d(self.out_dim * 4, self.out_dim * 8)
        self.pool_4_1 = maxpool()

        # Encoder (Modality 3) T1c
        self.down_1_2 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_2 = maxpool()
        self.down_2_2 = ConvBlock2d(self.out_dim, self.out_dim * 2)
        self.pool_2_2 = maxpool()
        self.down_3_2 = ConvBlock2d(self.out_dim * 2, self.out_dim * 4)
        self.pool_3_2 = maxpool()
        self.down_4_2 = ConvBlock2d(self.out_dim * 4, self.out_dim * 8)
        self.pool_4_2 = maxpool()

        # Encoder (Modality 4) T2
        self.down_1_3 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_3 = maxpool()
        self.down_2_3 = ConvBlock2d(self.out_dim, self.out_dim * 2)
        self.pool_2_3 = maxpool()
        self.down_3_3 = ConvBlock2d(self.out_dim * 2, self.out_dim * 4)
        self.pool_3_3 = maxpool()
        self.down_4_3 = ConvBlock2d(self.out_dim * 4, self.out_dim * 8)
        self.pool_4_3 = maxpool()

        # bridge between encoder decoder
        self.bridge = ConvBlock2d(self.out_dim * 32, self.out_dim * 16)

        # bifpn
        self.bifpn = BiFPNUnit(n=self.in_dim, channels=self.out_dim)

        # ~~ Decoding Path ~~#

        self.upLayer1 = UpBlock2d(self.out_dim * 16, self.out_dim * 8)
        self.upLayer2 = UpBlock2d(self.out_dim * 8, self.out_dim * 4)
        self.upLayer3 = UpBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.upLayer4 = UpBlock2d(self.out_dim * 2, self.out_dim * 1)

        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        # ~~~ Encoding Path ~~

        i0 = input[:, 0:1, :, :]  # comment to remove flair
        i1 = input[:, 1:2, :, :]  # comment to remove t1
        i2 = input[:, 2:3, :, :]  # comment to remove t1c
        i3 = input[:, 3:4, :, :]  # comment to remove t2
        print('i0')
        print(i0.shape)
        print('i1')
        print(i1.shape)
        print('i2')
        print(i2.shape)
        print('i3')
        print(i3.shape)

        down_1_0 = self.down_1_0(i0)
        down_1_1 = self.down_1_1(i1)
        down_1_2 = self.down_1_2(i2)
        down_1_3 = self.down_1_3(i3)
        print('down_1_0')
        print(down_1_0.shape)

        input_2nd_0 = self.pool_1_0(down_1_0)
        input_2nd_1 = self.pool_1_1(down_1_1)
        input_2nd_2 = self.pool_1_2(down_1_2)
        input_2nd_3 = self.pool_1_3(down_1_3)
        print('input_2nd_0')
        print(input_2nd_0.shape)

        down_2_0 = self.down_2_0(input_2nd_0)
        down_2_1 = self.down_2_1(input_2nd_1)
        down_2_2 = self.down_2_2(input_2nd_2)
        down_2_3 = self.down_2_3(input_2nd_3)
        print('down_2_0')
        print(down_2_0.shape)

        input_3rd_0 = self.pool_2_0(down_2_0)
        input_3rd_1 = self.pool_2_1(down_2_1)
        input_3rd_2 = self.pool_2_2(down_2_2)
        input_3rd_3 = self.pool_2_3(down_2_3)
        print('input_3rd_0')
        print(input_3rd_0.shape)

        down_3_0 = self.down_3_0(input_3rd_0)
        down_3_1 = self.down_3_1(input_3rd_1)
        down_3_2 = self.down_3_2(input_3rd_2)
        down_3_3 = self.down_3_3(input_3rd_3)
        print('down_3_0')
        print(down_3_0.shape)

        input_4th_0 = self.pool_3_0(down_3_0)
        input_4th_1 = self.pool_3_1(down_3_1)
        input_4th_2 = self.pool_3_2(down_3_2)
        input_4th_3 = self.pool_3_3(down_3_3)
        print('input_4th_0')
        print(input_4th_0.shape)

        down_4_0 = self.down_4_0(input_4th_0)  # 8C
        down_4_1 = self.down_4_1(input_4th_1)
        down_4_2 = self.down_4_2(input_4th_2)
        down_4_3 = self.down_4_3(input_4th_3)
        print('down_4_0')
        print(down_4_0.shape)

        down_4_0m = self.pool_4_0(down_4_0)
        down_4_1m = self.pool_4_0(down_4_1)
        down_4_2m = self.pool_4_0(down_4_2)
        down_4_3m = self.pool_4_0(down_4_3)
        print('down_4_0m')
        print(down_4_0m.shape)

        inputBridge = torch.cat((down_4_0m, down_4_1m, down_4_2m, down_4_3m), dim=1)
        print('inputBridge')
        print(inputBridge.shape)

        bridge = self.bridge(inputBridge)
        print('bridge ')
        print(bridge.shape)

        skip_1 = torch.cat((down_4_0, down_4_1, down_4_2, down_4_3), dim=1)
        print('skip_1 ')
        print(skip_1.shape)
        skip_2 = torch.cat((down_3_0, down_3_1, down_3_2, down_3_3), dim=1)
        print('skip_2 ')
        print(skip_2.shape)
        skip_3 = torch.cat((down_2_0, down_2_1, down_2_2, down_2_3), dim=1)
        print('skip_3 ')
        print(skip_3.shape)
        skip_4 = torch.cat((down_1_0, down_1_1, down_1_2, down_1_3), dim=1)
        print('skip_4 ')
        print(skip_4.shape)

        x12, x22, x32, x42 = self.bifpn(skip_4, skip_3, skip_2, skip_1)

        x = self.upLayer1(bridge, x42)
        # x = self.upLayer1(x42)
        print('uplayer1')
        print(x.shape)
        x = self.upLayer2(x, x32)
        print('uplayer2')
        print(x.shape)
        x = self.upLayer3(x, x22)
        print('uplayer3')
        print(x.shape)
        x = self.upLayer4(x, x12)
        print('uplayer4')
        print(x.shape)

        return self.out(x)

if __name__ == "__main__":
    batch_size = 1
    num_classes = 5  # one hot
    initial_kernels = 32

    net = MM_BiFPN(1, num_classes, initial_kernels)
    print("total parameter:" + str(netSize(net)))  # 2860,0325
    torch.save(net.state_dict(), 'model.pth')
    MRI = torch.randn(batch_size, 4, 64, 64)  # Batchsize, modal, height,

    if torch.cuda.is_available():
        net = net.cuda()
        MRI = MRI.cuda()

    segmentation_prediction = net(MRI)
    print(segmentation_prediction.shape)
