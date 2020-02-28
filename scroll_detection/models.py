import numpy as np
import glob
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F



#input: torch.Size([6, 112, 128])
#output: torch.Size([2, 1])
class SixLayerCNN(nn.module):
    def __init__(self):
        super(InitialConvModel, self).__init__()

        self.conv1 = nn.Conv2d(6,128,kernel_size=7,stride=2.padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLu(inplace=True)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        #self.fc_middle = nn.Linear(256*7*8, 256*7*8)

        self.lastpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.fc_relu = nn.ReLU()
        # num_classes, filters, 1x1
        self.deconv1 = nn.ConvTranspose2d(128, 10, kernel_size=7, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        out0_size = x.size()
        x, indices_0 = self.maxpool(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        out1_size = x.size()
        x, indices_1 = self.maxpool(x)

        x = self.relu3(self.bn3(self.conv3(x)))
        out2_size = x.size()
        x, indices_2 = self.maxpool(x)

        x = self.unpool(x, indices_2, output_size=out2_size)
        x = self.deconv3(x)

        x = self.unpool(x, indices_1, output_size=out1_size)
        x = self.deconv2(x)
        x = self.unpool(x, indices_0, output_size=out0_size)
        x = self.deconv1(x)
        x = self.lastpool(x)
        x = F.interpolate(x, size=[2, 1],mode='nearest')
        return x
