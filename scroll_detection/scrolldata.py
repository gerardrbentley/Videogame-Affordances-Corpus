import numpy as np
import glob
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

#collate function?


class ScrollingDataset(Dataset):
    def __init__(self, game='loz', data_dir='../games/', transform=None):
        self.image_dir = os.path.join(data_dir, game, 'screenshots')

        self.image_folders = next(os.walk(self.image_dir))[
                                  1]  # stop iteration error in os walk?
        self.length = len(self.image_folders)
        self.transform = transform

    def __len__(self):
        return self.length

    ## TODO: write a function that takes an image, assigns a random scroll and then transforms the image

    def __getitem__(self, idx):
        folder_name = self.image_folders[idx]
        screenshot_file = os.path.join(
                self.image_dir, folder_name, f'{folder_name}.png')

        image = Image.open(screenshot_file).convert(
            'RGB')  # image size = 256,224
        print("Image size: ", image.size)
        target = (0, 0)  # replace with random int
        #target should be a tuple of x,y
        sample = {'image': image, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


sd = ScrollingDataset()

sd.__getitem__(0)
