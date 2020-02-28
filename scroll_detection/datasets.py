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
    def __init__(self, game='loz', data_dir='../../games/', transform=None,max_shift=8):
        self.image_dir = os.path.join(data_dir, game, 'screenshots/')
        self.image_folders = next(os.walk(self.image_dir))[1]
        self.length = len(self.image_folders)
        self.transform = transform
        self.max_shift = max_shift

    def __len__(self):
        return self.length

    ## TODO: write a function that takes an image, assigns a random scroll and then transforms the image

    def __getitem__(self, idx):
        folder_name = self.image_folders[idx]
        screenshot_file = os.path.join(self.image_dir, folder_name, f'{folder_name}.png')
        image = Image.open(screenshot_file).convert('RGB')  # image size = 256,224
        x_dim , y_dim = image.size #x,y

        x_shift = random.randint(0, min(x_dim/2,self.max_shift))
        y_shift = random.randint(0, min(y_dim/2,self.max_shift))
        y_shift = 0 #TODO: remove later
        #crop image from (256, 224) to (128,112) upper lefthand corner (0,0,x_dim/2,y_dim/2)
        original_im = image.crop((0,0,(x_dim/2),(y_dim/2)))
        shifted_im = image.crop((x_shift,y_shift,(x_dim/2)+x_shift,(y_dim/2)+y_shift))
        original_im.show()
        shifted_im.show()
        image_tuple = (original_im,shifted_im)
        target = (x_shift,y_shift)
        sample = {'image': image_tuple, 'target': target}
        if self.transform:
            sample = self.transform(sample)
        return sample

class TwoImagesBCHW(object):
    """Convert two images into stacked Tensors for input to model
    in the form Batch x Channels x Height x Width """

    def __call__(self, sample):
        images, shift = list(sample['image']), sample['target']
        fn = transforms.ToTensor()
        images = list(map(fn, images))
        cat_ = torch.cat(images,0)
        shift = torch.tensor(shift).view((2,1))
        print(shift)
        return {'image': cat_, 'target': shift}

sd = ScrollingDataset(transform=transforms.Compose([ToTensorBCHW()]))

a = sd.__getitem__(0)
print(a["image"].shape,a["target"].shape)
