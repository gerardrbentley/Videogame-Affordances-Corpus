import numpy as np
import glob
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F


class SixLayerCNN(nn.module):
    def __init__(self):
        super(InitialConvModel, self).__init__()

        self.conv1 = nn.Conv2d(6,128)

    def forward(self, x):
