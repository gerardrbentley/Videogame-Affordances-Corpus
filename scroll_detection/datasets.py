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
import random

#collate function?


class ScrollingDataset(Dataset):
    def __init__(self, game='loz', data_dir='../../games/', transform=None):
        self.image_dir = os.path.join(data_dir, game, 'screenshots/')
        self.image_folders = next(os.walk(self.image_dir))[1]
        self.length = len(self.image_folders)
        self.transform = transform

    def __len__(self):
        return self.length

    ## TODO: write a function that takes an image, assigns a random scroll and then transforms the image

    def __getitem__(self, idx):
        folder_name = self.image_folders[idx]
        screenshot_file = os.path.join(self.image_dir, folder_name, f'{folder_name}.png')
        image = Image.open(screenshot_file).convert('RGB')  # image size = 256,224
        x_dim , y_dim = image.size #x,y

        x_shift = random.randint(0, x_dim/2)
        y_shift = 0
        #crop image from (256, 224) to (128,112) upper lefthand corner (0,0,x_dim/2,y_dim/2)
        original_im = image.crop((0,0,(x_dim/2),(y_dim/2)))
        shifted_im = image.crop((x_shift,0,(x_dim/2)+x_shift,(y_dim/2)))

        image_tuple = (original_im,shifted_im)
        target = (x_shift,y_shift)
        sample = {'image': image_tuple, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


sd = ScrollingDataset()
sd.__getitem__(0)
