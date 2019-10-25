import numpy as np
import glob
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

affords = ["solid", "movable", "destroyable",
           "dangerous", "gettable", "portal", "usable", "changeable", "ui"]
AFFORDANCES = affords

dum_img = torch.randn(3, 224, 256)
dum_label = torch.randn(9, 224, 256)


class DummyDataset(Dataset):
    def __init__(self):
        self.length = 960

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {'image': dum_img, 'affordances': dum_label}


class TensorAffordancesDataset(Dataset):
    def __init__(self, image_dir, affordances_dir, transform=None):
        self.image_dir = image_dir
        self.affordances_dir = affordances_dir

        image_files = glob.glob(image_dir + "*.pt")
        self.length = len(image_files)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label_file = self.affordances_dir + str(idx)+".pt"
        img_file = self.image_dir + str(idx)+".pt"
        stacked_tensor = torch.load(label_file)
        stacked_tensor.requires_grad_(False)
        image_tensor = torch.load(img_file)
        sample = {'image': image_tensor, 'affordances': stacked_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample


class SKAffordancesDataset(Dataset):
    def __init__(self, image_dir, affordances_dir, transform=None):
        self.image_dir = image_dir
        self.affordances_dir = affordances_dir

        image_files = glob.glob(image_dir + "*.pt")
        self.length = len(image_files)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label_file = self.affordances_dir + str(idx)+".pt"
        img_file = self.image_dir + str(idx)+".pt"
        stacked_tensor = torch.load(label_file)
        stacked_tensor.requires_grad_(False)
        image_tensor = torch.load(img_file)
        sample = {'image': image_tensor, 'affordances': stacked_tensor}

        if self.transform:
            sample = self.transform(sample)

        return (sample['image'], sample['affordances'])


class GrayAffordancesDataset(Dataset):
    def __init__(self, image_dir, affordances_dir, transform=None):
        self.image_dir = image_dir
        self.affordances_dir = affordances_dir

        image_files = glob.glob(image_dir + "*.png")
        self.length = len(image_files)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = Image.open(self.image_dir + str(idx)+".png").convert("L")
        label_file = self.affordances_dir + str(idx)+".pt"
        stacked_tensor = torch.load(label_file)
        stacked_tensor.requires_grad_(False)
        stacked_np = stacked_tensor.numpy()
        stacked_np = (stacked_np * 255).astype(np.uint8)
        first_channel = Image.fromarray(stacked_np[:, :, 0], mode="L")
        channels = (first_channel,)
        for x in range(1, 9):
            channel = stacked_np[:, :, x]
            channel_PIL = Image.fromarray(channel, mode='L')
            channels += (channel_PIL,)

        affordances = channels

        sample = {'image': image, 'affordances': affordances}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ImageAffordancesDataset(Dataset):
    def __init__(self, image_dir, affordances_dir, transform=None):
        self.image_dir = image_dir
        self.affordances_dir = affordances_dir

        image_files = glob.glob(image_dir + "*.png")
        self.length = len(image_files)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = Image.open(self.image_dir
                           + str(idx)+".png").convert('RGB')
        label_file = self.affordances_dir + str(idx)+".pt"
        stacked_tensor = torch.load(label_file)
        stacked_tensor.requires_grad_(False)
        stacked_np = stacked_tensor.numpy()
        stacked_np = (stacked_np * 255).astype(np.uint8)
        first_channel = Image.fromarray(stacked_np[:, :, 0], mode="L")
        channels = (first_channel,)
        for x in range(1, 9):
            channel = stacked_np[:, :, x]
            channel_PIL = Image.fromarray(channel, mode='L')
            channels += (channel_PIL,)

        affordances = channels

        sample = {'image': image, 'affordances': affordances}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """ output_size = (new_w, new_h) """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, affordances = sample['image'], sample['affordances']

        w, h = image.size

        new_w, new_h = self.output_size
        new_w, new_h = int(new_w), int(new_h)

        fn = transforms.Resize((new_h, new_w))
        image = fn(image)
        affordances = tuple(map(fn, affordances))
        return {'image': image, 'affordances': affordances}


class RandomCrop(object):
    """ output_size = (new_w, new_h) """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        assert len(output_size) == 2
        self.output_size = output_size

    def __call__(self, sample):
        image, affordances = sample['image'], sample['affordances']

        w, h = image.size
        new_w, new_h = self.output_size
        new_w, new_h = int(new_w), int(new_h)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        def fn(x): return x.crop([left, top, left+new_w, top+new_h])
        image = fn(image)
        affordances = tuple(map(fn, affordances))

        fn = transforms.Resize((h, w))
        image = fn(image)
        affordances = tuple(map(fn, affordances))
        return {'image': image, 'affordances': affordances}


class RandomRotate(object):
    """ angle = max degress counter clockwise """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, affordances = sample['image'], sample['affordances']
        rand_deg = np.random.randint(0, self.angle)
        def fn(x): return x.rotate(rand_deg, resample=Image.BILINEAR)
        image = fn(image)
        affordances = tuple(map(fn, affordances))

        return {'image': image, 'affordances': affordances}


class RandomFlip(object):
    """default flips Left right"""

    def __init__(self, flip_left_right=True, flip_top_bottom=False):
        self.lr = flip_left_right
        self.tb = flip_top_bottom

    def __call__(self, sample):
        image, affordances = sample['image'], sample['affordances']
        rand = np.random.randint(0, 1)
        if self.lr and rand:
            def fn(x): return x.transpose(Image.FLIP_LEFT_RIGHT)
            image = fn(image)
            affordances = tuple(map(fn, affordances))
        rand = np.random.randint(0, 1)
        if self.tb and rand:
            def fn(x): return x.transpose(Image.FLIP_TOP_BOTTOM)
            image = fn(image)
            affordances = tuple(map(fn, affordances))

        return {'image': image, 'affordances': affordances}


class RandomShear(object):
    """deg(float) max 180. random positive or negative value below"""

    def __init__(self, deg):
        self.deg = deg

    def __call__(self, sample):
        image, affordances = sample['image'], sample['affordances']
        rand_deg = np.random.randint(-self.deg, self.deg)
        def fn(x): return transforms.functional.affine(x, angle=0, translate=[
               0, 0], scale=1, shear=rand_deg, resample=Image.BILINEAR)
        image = fn(image)
        affordances = tuple(map(fn, affordances))

        return {'image': image, 'affordances': affordances}


class RandomReColor(object):
    """
        From torchvision.transforms
        brightness (float or tuple of python:float (min, max)) – How much to jitter brightness. brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers.

        contrast (float or tuple of python:float (min, max)) – How much to jitter contrast. contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers.

        saturation (float or tuple of python:float (mGammaChangein, max)) – How much to jitter saturation. saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers.

        hue (float or tuple of python:float (min, max)) – How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.

"""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        image, affordances = sample['image'], sample['affordances']
        image = transforms.ColorJitter(
            self.brightness, self.contrast, self.saturation, self.hue)(image)

        return {'image': image, 'affordances': affordances}


class GammaChange(object):
        def __init__(self, gamma, gain=1):
            self.gamma = gamma
            self.gain = gain

        def __call__(self, sample):
            image, affordances = sample['image'], sample['affordances']
            image = transforms.functional.adjust_gamma(
                image, self.gamma, self.gain)
            return {'image': image, 'affordances': affordances}


class ToTensorBCHW(object):
    """Convert image and affordance to Tensors for input to model
    in the form Batch x Channels x Height x Width """

    def __call__(self, sample):
        image, affordances = sample['image'], sample['affordances']
        fn = transforms.ToTensor()
        affordances = list(map(fn, affordances))
        stack = torch.stack(affordances, dim=1)
        stack.requires_grad_(False)
        stack = stack.squeeze(0)

        image = fn(image)
        image.requires_grad_(False)
        return {'image': image, 'affordances': stack}


def save_model(network, title):
    path = "trained_models/" + title
    print("saving: ", path)
    torch.save(network.state_dict(), path)


def load_model(network, title):
    path = "trained_models/" + title
    network.load_state_dict(torch.load(path))


multitransform = transforms.Compose([RandomCrop((180, 180)), RandomFlip(
    flip_left_right=True, flip_top_bottom=True), RandomReColor(hue=0.3, saturation=0.3, contrast=0.2), ToTensorBCHW()])
recolortransform = transforms.Compose(
    [RandomReColor(hue=0.3, contrast=0.2, saturation=0.3), ToTensorBCHW()])


#tests for tranforms
def test():
    writer = SummaryWriter()

    #data = GrayAffordancesDataset(image_dir="data/train_img/", affordances_dir="data/train_label/")
    data = TensorAffordancesDataset(
        image_dir="data/aug_img/", affordances_dir="data/aug_label/")

    test = transforms.Compose([ToTensorBCHW()])
    color = transforms.Compose(
        [RandomReColor(hue=0.5, saturation=0.3), ToTensorBCHW()])
    color2 = transforms.Compose(
        [RandomReColor(hue=0.5, contrast=0.1, saturation=0.3), ToTensorBCHW()])
    all = transforms.Compose([RandomCrop((150, 100)), RandomFlip(flip_left_right=True, flip_top_bottom=True), RandomShear(
        deg=50), RandomRotate(330), RandomReColor(hue=0.5, saturation=0.3, contrast=0.2), ToTensorBCHW()])
    sample = data[3]

    for i in range(0, 400, 7):
        sample = data[i]
        image, affordances = sample['image'], sample['affordances'].squeeze(0)
        for x in range(9):
            writer.add_image('screenshot_'+str(i)+'/'+affords[x],
                             affordances[x, :, :], 0, dataformats='HW')

        writer.add_image('screenshot_'+str(i)+'/' + 'a',
                         image, 0, dataformats='CHW')
    # for i, tsfrm in enumerate([all, color, test]):
    #     for z in range(1):
    #         tsample = tsfrm(sample)
    #         #tsample = tensor(tsample)
    #         #channels_actual = []
    #         # for x in range(11):
    #         #     channel = sample['affordance'][:, :,x]
    #         #     channels_actual.append(channel)
    #         #     writer.add_image('targetmaps_'+str(i)+'/'+str(x),
    #         #                      channels_actual[x], 0, dataformats='HW')
    #         channels = tsample['affordances'].squeeze(0)
    #         #print(channels.size())
    #         img = tsample['image'].squeeze(0)
    #         for x in range(9):
    #             writer.add_image('screenshot_'+str(i)+'/'+affords[x],
    #                              channels[x,:,:], 0, dataformats='HW')
    #
    #         # writer.add_image('screenshot_tra'+'/'+str(i),
    #         #                  channels[0,:,:], 0, dataformats='HW')
    #         writer.add_image('screenshot_'+str(i)+'/' +str(z),
    #                   img, 0, dataformats='HW')

    writer.close()

    print('done 64')


def pre_augment():
    # mode = 'train'
    #
    # if mode == 'train':
    #     img_dir = 'data/train_img/'
    #     label_dir = 'data/train_label/'
    # elif mode == 'test':
    #     img_dir = 'data/test_img/'
    #     label_dir = 'data/test_label/'

    # data = GrayAffordancesDataset(image_dir=img_dir, affordances_dir=label_dir, transform=transforms.Compose([ToTensorBCHW()]))
    # test_dataset = GrayAffordancesDataset(image_dir="data/test_img/", affordances_dir="data/test_label/")
    # train_dataset = GrayAffordancesDataset(image_dir="data/train_img/", affordances_dir="data/train_label/")
    # tagged_data = ConcatDataset([test_dataset, train_dataset])
    tagged_data = GrayAffordancesDataset(
        image_dir="data/loz/img/", affordances_dir="data/loz/label/")
    print(len(tagged_data))

    # justcrop = GrayAffordancesDataset(image_dir=img_dir, affordances_dir=label_dir, transform=transforms.Compose([RandomCrop((96,96)),RandomFlip(flip_left_right=True, flip_top_bottom=True), ToTensorBCHW()]))
    # justrotate = GrayAffordancesDataset(image_dir=img_dir, affordances_dir=label_dir, transform=transforms.Compose([RandomRotate(350), RandomFlip(flip_left_right=True, flip_top_bottom=True), ToTensorBCHW()]))
    # smallcroprotateflip = GrayAffordancesDataset(image_dir=img_dir, affordances_dir=label_dir, transform=transforms.Compose([RandomCrop((64,64)), RandomFlip(flip_left_right=True, flip_top_bottom=True), RandomRotate(350), ToTensorBCHW()]))
    rotate = transforms.Compose([RandomCrop((224, 208)), RandomRotate(
        350), RandomFlip(flip_left_right=True, flip_top_bottom=True), ToTensorBCHW()])
    crop = transforms.Compose([RandomCrop((200, 190)), RandomFlip(
        flip_left_right=True, flip_top_bottom=True), ToTensorBCHW()])
    input_transform = ToTensorBCHW()

    ctr = 0
    for y in range(len(tagged_data)):
        sample = input_transform(tagged_data[y])
        image, affordances = sample['image'], sample['affordances']
        torch.save(image, 'data/aug_img/'+str(ctr)+'.pt')
        torch.save(affordances, 'data/aug_label/'+str(ctr)+'.pt')
        ctr += 1
    for x in range(200):
        print('augmenting round {:d}, ctr{:d}'.format(x, ctr))
        for y in range(len(tagged_data)):
            sample = crop(tagged_data[y])
            image, affordances = sample['image'], sample['affordances']
            torch.save(image, 'data/aug_img/'+str(ctr)+'.pt')
            torch.save(affordances, 'data/aug_label/'+str(ctr)+'.pt')
            ctr += 1
            if ctr == 2400:
                break
            sample = rotate(tagged_data[y])
            image, affordances = sample['image'], sample['affordances']
            torch.save(image, 'data/aug_img/'+str(ctr)+'.pt')
            torch.save(affordances, 'data/aug_label/'+str(ctr)+'.pt')
            ctr += 1
            if ctr == 2400:
                break
        if ctr == 2400:
            break

    print('done augmenting')


if __name__ == '__main__':
    pre_augment()
    test()
