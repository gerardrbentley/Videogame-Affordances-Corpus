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

affords = ["solid", "movable", "destroyable",
           "dangerous", "gettable", "portal", "usable", "changeable", "ui", "permeable"]
AFFORDANCES = affords

dum_img = torch.randn(3, 224, 256)
dum_label = torch.randn(10, 224, 256)


class DummyDataset(Dataset):
    def __init__(self):
        self.length = 960

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {'image': dum_img, 'target': dum_label}


class GameAffordancesDataset(Dataset):
    def __init__(self, game='loz', data_dir='/app/games/', transform=None):
        self.image_dir = os.path.join(data_dir, game, 'screenshots')

        self.image_folders = next(os.walk(self.image_dir))[1]
        self.length = len(self.image_folders)
        self.transform = transform

    def __len__(self):
        return self.length

    def get_affordances_target(self, folder_name):
        label_files = glob.glob(os.path.join(
                self.image_dir, folder_name, "*.npy"))
        if len(label_files) > 0:
            label_file = label_files[0]

            stacked_np = np.load(label_file)
            stacked_np = (stacked_np * 255).astype(np.uint8)
            first_channel = Image.fromarray(stacked_np[:, :, 0], mode="L")
            channels = (first_channel,)
            for x in range(1, 10):
                channel = stacked_np[:, :, x]
                channel_PIL = Image.fromarray(channel, mode='L')
                channels += (channel_PIL,)

            affordances = channels

        return affordances

    def __getitem__(self, idx):
        folder_name = self.image_folders[idx]
        screenshot_file = os.path.join(
                self.image_dir, folder_name, f'{folder_name}.png')

        image = Image.open(screenshot_file).convert('RGB')
        target = self.get_affordances_target(folder_name)

        sample = {'image': image, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """ output_size = (new_w, new_h) """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, affordances = sample['image'], sample['target']

        w, h = image.size

        new_w, new_h = self.output_size
        new_w, new_h = int(new_w), int(new_h)

        fn = transforms.Resize((new_h, new_w))
        image = fn(image)
        affordances = tuple(map(fn, affordances))
        return {'image': image, 'target': affordances}


class RandomCrop(object):
    """ output_size = (new_w, new_h) """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        assert len(output_size) == 2
        self.output_size = output_size

    def __call__(self, sample):
        image, affordances = sample['image'], sample['target']

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
        return {'image': image, 'target': affordances}


class RandomRotate(object):
    """ angle = max degress counter clockwise """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, affordances = sample['image'], sample['target']
        rand_deg = np.random.randint(0, self.angle)
        def fn(x): return x.rotate(rand_deg, resample=Image.BILINEAR)
        image = fn(image)
        affordances = tuple(map(fn, affordances))

        return {'image': image, 'target': affordances}


class RandomFlip(object):
    """default flips Left right"""

    def __init__(self, flip_left_right=True, flip_top_bottom=False):
        self.lr = flip_left_right
        self.tb = flip_top_bottom

    def __call__(self, sample):
        image, affordances = sample['image'], sample['target']
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

        return {'image': image, 'target': affordances}


class RandomShear(object):
    """deg(float) max 180. random positive or negative value below"""

    def __init__(self, deg):
        self.deg = deg

    def __call__(self, sample):
        image, affordances = sample['image'], sample['target']
        rand_deg = np.random.randint(-self.deg, self.deg)
        def fn(x): return transforms.functional.affine(x, angle=0, translate=[
               0, 0], scale=1, shear=rand_deg, resample=Image.BILINEAR)
        image = fn(image)
        affordances = tuple(map(fn, affordances))

        return {'image': image, 'target': affordances}


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
        image, affordances = sample['image'], sample['target']
        image = transforms.ColorJitter(
            self.brightness, self.contrast, self.saturation, self.hue)(image)

        return {'image': image, 'target': affordances}


class GammaChange(object):
        def __init__(self, gamma, gain=1):
            self.gamma = gamma
            self.gain = gain

        def __call__(self, sample):
            image, affordances = sample['image'], sample['target']
            image = transforms.functional.adjust_gamma(
                image, self.gamma, self.gain)
            return {'image': image, 'target': affordances}


class ToTensorBCHW(object):
    """Convert image and affordance to Tensors for input to model
    in the form Batch x Channels x Height x Width """

    def __call__(self, sample):
        image, affordances = sample['image'], sample['target']
        fn = transforms.ToTensor()
        affordances = list(map(fn, affordances))
        stack = torch.stack(affordances, dim=1)
        stack.requires_grad_(False)
        stack = stack.squeeze(0)

        image = fn(image)
        image.requires_grad_(False)
        return {'image': image, 'target': stack}


def save_model(network, title):
    path = "trained_models/" + title
    print("saving: ", path)
    torch.save(network.state_dict(), path)


def load_model(network, title):
    path = "trained_models/" + title
    network.load_state_dict(torch.load(path))

def torch_max_min_str(tensor):
    mini = torch.min(tensor).item()
    maxi = torch.max(tensor).item()
    return f'Range: {mini} - {maxi}'

multitransform = transforms.Compose([RandomCrop((180, 180)), RandomFlip(
    flip_left_right=True, flip_top_bottom=True), RandomReColor(hue=0.3, saturation=0.3, contrast=0.2), ToTensorBCHW()])
recolortransform = transforms.Compose(
    [RandomReColor(hue=0.3, contrast=0.2, saturation=0.3), ToTensorBCHW()])


#tests for tranforms
def test():
    testtransform = transforms.Compose([ToTensorBCHW()])
    data = GameAffordancesDataset(game='loz', data_dir='../games')
    print(len(data))
    print(type(data[0]))
    print(type(data[0]['image']), type(data[0]['target']))
    x = testtransform(data[0])
    img, targ = x['image'], x['target']
    print(type(img), type(targ))
    print((img).shape, (targ).shape)
    print(torch_max_min_str(img), torch_max_min_str(targ))


    print('done 47')


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
    # pre_augment()
    test()
