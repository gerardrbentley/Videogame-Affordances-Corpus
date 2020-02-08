from datasets import ToTensorBCHW, ScrollingDataset
from models import InitialConvModel

from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
import torch.nn as nn
import torch
import argparse
import time
import datetime
import os
import shutil
import sys
import uuid


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        input_transform = transforms.Compose([ToTensorBCHW()])
