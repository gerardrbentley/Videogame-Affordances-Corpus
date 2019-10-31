
import cv2
import numpy as np
import dask
from dask.diagnostics import ProgressBar
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt

import os
import argparse
import pickle

from yolo.models import Darknet
from yolo.utils.utils import load_classes, rescale_boxes, non_max_suppression
from yolo.utils.datasets import resize, pad_to_square

from pre_process_utils import is_unique_by_mse, show_images, myint


def detect_sprites(image, args):
    """
    Parameters
    ----------
    image: RGB opencv image (ndarray)
    args: must include conf_thres, nms_thres, yolo_model, (if verbose then  grid_size, classes)

    Returns
    ---------
    (image, sprite_locs) where
        image: input image with sprite bounding boxes drawn onto it
        sprite_locs: list of locations in format (left, top, right, bot)
    """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    height, width, *_ = image.shape

    yolo_pil = Image.fromarray(image)
    # print(type(yolo_pil))
    yolo_input = ToTensor()(yolo_pil)
    yolo_input, _ = pad_to_square(yolo_input, 0)
    yolo_input = resize(yolo_input, 416).unsqueeze(0)
    yolo_input = Variable(yolo_input.type(Tensor))

    # Get detections
    detections = []
    with torch.no_grad():
        output = args.yolo_model(yolo_input)
        # print('detect: ', output.shape)
        output = non_max_suppression(
            output, args.conf_thres, args.nms_thres)
        detections.extend(output)

    # Format detected bboxes as (left, top, right, bot)
    sprite_locs = []

    #Detections batched by default, but we only have batch of 1
    if detections[0] is not None:
        # Rescale boxes to original image
        detections[0] = rescale_boxes(
            detections[0], 416, np.array(yolo_pil.convert('L')).shape[:2])

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
            #If not right next to edge, relax bounding box by 1 pixel in each direction
            relax_factor = 1
            if myint(x1) < relax_factor or myint(y1) < relax_factor or width - myint(x2) < relax_factor or height - myint(y2) < relax_factor:
                print('EDGE Avoided')
                left = myint(x1)
                right = myint(x2)
                top = myint(y1)
                bot = myint(y2)
            else:
                left = myint(x1) - relax_factor
                right = myint(x2) + relax_factor
                top = myint(y1) - relax_factor
                bot = myint(y2) + relax_factor

            sprite_locs.append((left, top, right, bot))
            cv2.rectangle(image, (left, top), (right, bot),
                          color=(0, 255, 0), thickness=2)

            if args.verbose:
                # rows and cols of detection in original image
                leftcol = left // args.grid_size
                rightcol = right // args.grid_size
                toprow = top // args.grid_size
                botrow = bot // args.grid_size

                print("\t+ Label: %s, Conf: %.5f," %
                      (args.classes[int(cls_pred)], cls_conf.item()))
                print(f'\t\t+ {leftcol}, {rightcol}, {toprow}, {botrow}')

    return image, sprite_locs


@dask.delayed
def get_unique_tiles(image, offset_y=0, offset_x=0, grid_size=16, ignore=[]):
    """
    Parameters
    ----------
    image: full RGB opencv image (ndarray). pass copy as it is cropped
    offset_y, offset_x: Where to start crop of image. int < grid_size.
    grid_size: how big of tiles to make
    ignore: list of locations (left, top, right, bot) to be ignored.
        translated into rows and cols of cropped image

    Returns
    ---------
    (encoded_tiles, offset_y, offset_x, num_ignored_indices) where
        encoded_tiles: list of unique tiles in image at offset as encoded pngs (1d ndarray)
        offset_y, offset_x: input offsets
        num_ignored_indices: number of (r,c) locations covered by sprites in input ignore list
    """
    h, w, *_ = image.shape

    # Need to treat 0,0 offset the same as others, else it will observe one more row and col
    xmax = w-grid_size + offset_x
    cols = (w-grid_size)//grid_size
    ymax = h-grid_size + offset_y
    rows = (h-grid_size)//grid_size

    cropped_img = image[offset_y:ymax, offset_x:xmax]

    unique_tiles = []
    num_dupes = 0

    # Get rows and cols that are covered by sprites in `ignore` list
    ignore_idxs = []
    if len(ignore) != 0:
        for (x1, y1, x2, y2) in ignore:
            left = x1 - offset_x
            right = x2 - offset_x
            top = y1 - offset_y
            bot = y2 - offset_y

            leftcol = left // args.grid_size
            rightcol = right // args.grid_size
            toprow = top // args.grid_size
            botrow = bot // args.grid_size

            if args.verbose:
                print(f'\t\t+ {leftcol}, {rightcol}, {toprow}, {botrow}')

            # include last value
            for r in range(toprow, botrow+1):
                for c in range(leftcol, rightcol+1):
                    ignore_idxs.append((r, c))

    no_sprite_indices = [(r, c) for r in range(rows)
                         for c in range(cols) if (r, c) not in ignore_idxs]

    # crop out each potential tile and add to list if not already included
    for (r, c) in no_sprite_indices:
        r_idx = r*grid_size
        c_idx = c*grid_size
        curr_tile = np.copy(cropped_img)[
                            r_idx:r_idx+grid_size, c_idx:c_idx+grid_size]

        if is_unique_by_mse(curr_tile, unique_tiles):
            unique_tiles.append(curr_tile)
        else:
            num_dupes += 1

    out_tiles = []
    for np_tile in unique_tiles:
        _, encoded_tile = cv2.imencode('.png', np_tile)
        out_tiles.append(encoded_tile)
    return (out_tiles, offset_y, offset_x, len(ignore_idxs))


def unique_tiles_all_offsets(args):
    """
    Parameters
    ----------
    args: Must include
        file: file path to load image
        ui_position: 'top' or 'bot'
        ui_height: int num pixels to crop for ui
        grid_size: int
        conf_thres, nms_thres, yolo_model: thresholds and pytorch yolo model for detection
        (if verbose then classes)

    Returns
    ---------
    List of (encoded_tiles, offset_y, offset_x, num_ignored_indices) for all offsets where
        encoded_tiles: list of unique tiles in image at given offset in encoded png bytes
        offset_y, offset_x: offsets for a given set of tiles
        num_ignored_indices: number of (r,c) locations covered by sprites for given offset
    """
    orig_image = cv2.imread(args.file, cv2.IMREAD_UNCHANGED)
    if orig_image.shape[2] == 4:
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGRA2BGR)
    # orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    if args.ui_position == 'top':
        orig_image = orig_image[args.ui_height:, :, :]
    else:
        h = orig_image.shape[0]
        orig_image = orig_image[:h-args.ui_height, :, :]

    # Yolo detect sprites bounding boxes
    if args is not None:
        sprited_image, sprite_locs = detect_sprites(
            np.copy(orig_image), args)
        torch.cuda.empty_cache()

    # print(type(image))
    potential_offsets = [(y, x) for x in range(0, args.grid_size)
                         for y in range(0, args.grid_size)]
    # potential_offsets = [(0, 0), (1, 0)]
    out = []
    for (y, x) in potential_offsets:
        out.append(get_unique_tiles(np.copy(orig_image),
                                    y, x, grid_size=args.grid_size, ignore=sprite_locs))

    # dask.visualize(out, filename='offsets.svg')
    res = dask.compute(out)

    if args.visualize:
        plt.figure()
        plt.imshow(sprited_image)

        # Needs np versions of tiles, but already encoded
        # show_images(res[0][1][4], cols=7)
        plt.pause(0.0001)
        plt.show()
    return res[0]


def len_unique_tiles(input):
    return len(input[0])


def k_best_sets(tile_sets_with_offsets, k):
    out_sort = sorted(tile_sets_with_offsets,
                      key=lambda x: len_unique_tiles(x))
    out = []
    for i in range(k):
        item = out_sort[i]
        out.append((item[0], item[1], item[2], item[3]))
    return out


def parse_args():
    parser = argparse.ArgumentParser(description='grid detection')

    parser.add_argument('--file', type=str, default='./0.png')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--game', type=str, default='sm3')
    parser.add_argument('--dest', type=str, default='output')
    parser.add_argument('--k', type=int,
                        default=5, help='num tile sets to select')
    parser.add_argument('--grid-size', type=int,
                        default=16, help='grid square size')
    parser.add_argument('--ui-height', type=int,
                        default=40, help='ignore this range')
    parser.add_argument('--ui-position', type=str,
                        default='bot', help='ui top or bot')
    parser.add_argument("--model_def", type=str,
                        default="yolo/config/yolov3-vg.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str,
                        default="yolo/checkpoints/yolov3_custom_ckpt_18.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str,
                        default="yolo/data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float,
                        default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4,
                        help="iou thresshold for non-maximum suppression")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    file_name = os.path.split(args.file)[1]
    file_name = os.path.splitext(file_name)[0]
    print(f'file: {args.file} - uuid {file_name}')
    pbar = ProgressBar()
    pbar.register()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    os.makedirs(args.dest, exist_ok=True)

    # Set up model
    model = Darknet(args.model_def, img_size=416).to(device)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()  # Set in evaluation mode
    args.yolo_model = model
    args.classes = load_classes(args.class_path)

    #Gen unique tile sets at all offsets
    tile_sets = unique_tiles_all_offsets(args)

    #Prune best k sets
    res = k_best_sets(tile_sets, args.k)

    os.makedirs(f'{args.dest}/{args.game}', exist_ok=True)
    pickle.dump(
        res, open(f'{args.dest}/{args.game}/{args.game}_{file_name}.tiles', 'wb'))

    # print(len(res))
    tset_lens = []
    y_offsets = []
    x_offsets = []
    num_ignored_tiles = []
    for i in res:
        tset_lens.append(len(i[0]))
        y_offsets.append((i[1]))
        x_offsets.append((i[2]))
        num_ignored_tiles.append((i[3]))
    df = pd.DataFrame({"tile_set len": tset_lens,
                       "y_offset": y_offsets, "x_offset": x_offsets, "ignored_tiles": num_ignored_tiles})
    df.to_csv(f'{args.dest}/{args.game}/{args.game}_{file_name}.csv')
