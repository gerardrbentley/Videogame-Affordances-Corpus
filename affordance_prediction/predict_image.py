
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from models import InitialConvModel
import affordancedata
from PIL import Image

import glob
import os
import argparse

TRAINED_MODELS = 'trained_models/'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Affordance Prediction Trial')
    # evaluation only
    parser.add_argument('--trial-id', type=str, default=None,
                        help='path to trained model')
    parser.add_argument('--image-path', type=str, default='data/validation_img/0.png',
                        help='path to test image')

    parser.add_argument('--output-dir', default='data/validation_output',
                        help='Directory for saving prediction images')

    parser.add_argument('--workers', '-j', type=int, default=16,
                        help='dataloader threads')

    args = parser.parse_args()

    return args


def main(args):
    #Test model

    print('Testing model')
    model = InitialConvModel()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    model.to('cpu')
    load_model(model, os.path.join(TRAINED_MODELS, args.trial_id))
    model.eval()

    affords = affordancedata.AFFORDANCES

    img = Image.open(args.image_path).convert('RGB')
    img.save(os.path.join(args.output_dir, 'test_img.png'))
    print('saved image copy to output dir')
    img = transforms.ToTensor()(img).unsqueeze(0)

    predictions = model(img)
    predictions = torch.sigmoid(predictions)

    for x in range(10):
        channel = predictions[0, x, :, :].detach().mul(255).numpy()
        #print(channel.size())
        #print('chan', affords[x], 'max' ,channel.max(), 'min', channel.min())
        out_img = Image.fromarray(channel)
        #Save PIL image
        if out_img.mode != 'RGB':
            out_img = out_img.convert('RGB')
        file_name = f'predict_{affords[x]}.png'
        out_img.save(os.path.join(args.output_dir, file_name))
        # writer.add_image('targetmaps_'+str(i)+'/'+affords[x],
        #                  channel, 0, dataformats='HW')


def save_model(network, path):
    print("saving model to: ", path)
    torch.save(network.state_dict(), path)


def load_model(network, path):
    print('loading model from: ', path)
    network.load_state_dict(torch.load(path))


def combine(a, b):
    out = torch.cat((a, b), 1)

    out = out.view(224, -1)
    return out


if __name__ == '__main__':
    args = parse_args()
    main(args)
