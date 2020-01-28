import numpy as np
import glob
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialConvModel(nn.Module):
    def __init__(self):
        super(InitialConvModel, self).__init__()

        # 1 input channels
        self.conv1 = nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)
        # self.fc1 = nn.Linear(6*224*256, 360)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        #self.fc_middle = nn.Linear(256*7*8, 256*7*8)

        self.lastpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.fc_relu = nn.ReLU()
        # num_classes, filters, 1x1
        self.deconv1 = nn.ConvTranspose2d(
            128, 10, kernel_size=7, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=5, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        #self.fc = nn.Linear(224*256*11, 224*256*11)
        # self.conv_last = nn.Conv2d(11, 11, 1, 1, 0)

    #@profile
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
        x = F.interpolate(x, size=[224, 256],
                          mode='nearest')

        #del out1_size, out2_size, indices, indices_2
        return x


class FourConvModel(nn.Module):
    def __init__(self):
        super(FourConvModel, self).__init__()

        # 1 input channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        # self.fc1 = nn.Linear(6*224*256, 360)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=1, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        #self.fc_middle = nn.Linear(64*7*8, 64*7*8)

        self.lastpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.fc_relu = nn.ReLU()
        # num_classes, filters, 1x1
        self.deconv1 = nn.ConvTranspose2d(
            64, 10, kernel_size=7, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(
            64, 64, kernel_size=7, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=7, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(
            128, 128, kernel_size=7, stride=1, padding=1)
        #self.fc = nn.Linear(224*256*11, 224*256*11)
        # self.conv_last = nn.Conv2d(11, 11, 1, 1, 0)

    #@profile
    def forward(self, x):

        x = self.relu1(self.bn1(self.conv1(x)))
        # out0_size = x.size()
        # x, indices_0 = self.maxpool(x)

        x = self.relu2(self.bn2(self.conv2(x)))
        out1_size = x.size()
        x, indices_1 = self.maxpool(x)

        x = self.relu3(self.bn3(self.conv3(x)))
        out2_size = x.size()
        x, indices_2 = self.maxpool(x)

        x = self.relu4(self.bn4(self.conv4(x)))
        out3_size = x.size()
        x, indices_3 = self.maxpool(x)

        x = self.unpool(x, indices_3, output_size=out3_size)
        x = self.bn4(self.deconv4(x))
        x = self.unpool(x, indices_2, output_size=out2_size)
        x = self.bn2(self.deconv3(x))

        x = self.unpool(x, indices_1, output_size=out1_size)
        x = self.bn1(self.deconv2(x))
        # x = self.unpool(x, indices_0, output_size=out0_size)
        x = (self.deconv1(x))
        x = self.lastpool(x)
        x = F.interpolate(x, size=[224, 256],
                          mode='nearest')

        return x


class ThreeConvModel(nn.Module):
    def __init__(self):
        super(ThreeConvModel, self).__init__()

        # 1 input channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        # self.fc1 = nn.Linear(6*224*256, 360)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=1, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        #self.fc_middle = nn.Linear(64*7*8, 64*7*8)

        self.lastpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.fc_relu = nn.ReLU()
        # num_classes, filters, 1x1
        self.deconv1 = nn.ConvTranspose2d(
            64, 10, kernel_size=7, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(
            64, 64, kernel_size=7, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(
            64, 64, kernel_size=7, stride=1, padding=1)
        #self.fc = nn.Linear(224*256*11, 224*256*11)
        # self.conv_last = nn.Conv2d(11, 11, 1, 1, 0)

    #@profile
    def forward(self, x):

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        out1_size = x.size()
        x, indices = self.maxpool(x)

        x = self.relu3(self.bn3(self.conv3(x)))
        out2_size = x.size()
        x, indices_2 = self.maxpool(x)

        x = self.unpool(x, indices_2, output_size=out2_size)
        x = self.deconv3(x)

        x = self.unpool(x, indices, output_size=out1_size)
        x = self.deconv2(x)
        x = self.deconv1(x)
        x = self.lastpool(x)
        x = F.interpolate(x, size=[224, 256],
                          mode='nearest')

        return x
