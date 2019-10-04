
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


def mse(a, b):
    diffs = np.square(np.subtract(a, b))
    total_diff = np.sum(diffs)
    return np.divide(total_diff, (a.shape[0] * a.shape[1]))


def show_images(images, cols=1, titles=None):
    """
    src: https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(figsize=(10, 10))
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        x = plt.imshow(image)
        # a.set_title(title)
        x.axes.get_xaxis().set_visible(False)
        x.axes.get_yaxis().set_visible(False)
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.axis('off')
    # plt.show()
    return fig


def is_unique_by_mse(new_tile, prev_tiles):
    for old_tile in prev_tiles:
        err = mse(new_tile, old_tile)
        # print('SSIM: {:.2f}'.format(similarity))
        if err < 0.00001:
            return False

    return True


def myint(torchscalar):
    return int(torchscalar.item())


def detect_sprites(image, args):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    height, width, *_ = image.shape

    yolo_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
    covered_grid_indices = []
    sprite_locs = []
    if detections[0] is not None:
        # Rescale boxes to original image
        detections[0] = rescale_boxes(
            detections[0], 416, np.array(yolo_pil.convert('L')).shape[:2])
        unique_labels = detections[0][:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:

            # cv2.rectangle(image, (int(x1.item()), int(y1.item())),
            #               (int(x2.item()), int(y2.item())), color=(0, 255, 0), thickness=2)

            relax_factor = 1
            if myint(x1) < relax_factor or myint(y1) < relax_factor or width - myint(x2) < relax_factor or height - myint(y2) < relax_factor:
                print('EDGE Avoided')
                # temp_crop = orig_PIL.convert('L').crop(
                #     box=(myint(x1), myint(y1), myint(x2), myint(y2)))
                left = myint(x1)
                right = myint(x2)
                top = myint(y1)
                bot = myint(y2)
            else:
                left = myint(x1) - relax_factor
                right = myint(x2) + relax_factor
                top = myint(y1) - relax_factor
                bot = myint(y2) + relax_factor
            cv2.rectangle(image, (left, top), (right, bot),
                          color=(0, 255, 0), thickness=2)
            leftcol = left // args.grid_size
            rightcol = right // args.grid_size
            toprow = top // args.grid_size
            botrow = bot // args.grid_size

            sprite_locs.append((left, top, right, bot))

            if args.verbose:
                print("\t+ Label: %s, Conf: %.5f," %
                      (args.classes[int(cls_pred)], cls_conf.item()))
                print(f'\t\t+ {leftcol}, {rightcol}, {toprow}, {botrow}')

            for r in range(toprow, botrow+1):
                for c in range(leftcol, rightcol+1):
                    covered_grid_indices.append((r, c))
            # covered_grid_indices.append((toprow, rightcol))
            # covered_grid_indices.append((botrow, rightcol))
            # covered_grid_indices.append((botrow, leftcol))
            # temp_crop = orig_PIL.convert('RGBA').crop(box=(myint(
            #     x1)-relax_factor, myint(y1)-relax_factor, myint(x2)+relax_factor, myint(y2)+relax_factor))
        # plt.figure()
        # a = plt.imshow(image)
        # plt.show()
    if args.verbose:
        print(len(covered_grid_indices), len(set(covered_grid_indices)))
    return image, set(covered_grid_indices), sprite_locs


@dask.delayed
def get_unique_tiles(image, offset_y=0, offset_x=0, grid_size=16, ignore=[]):
    h, w, c = image.shape
    # if offset_x == 0:
    #     xmax = w
    #     cols = (w)//grid_size
    # else:
    #     xmax = w-grid_size + offset_x
    #     cols = (w-grid_size)//grid_size
    # if offset_y == 0:
    #     ymax = h
    #     rows = (h)//grid_size
    # else:
    #     ymax = h-grid_size + offset_y
    #     rows = (h-grid_size)//grid_size
    xmax = w-grid_size + offset_x
    cols = (w-grid_size)//grid_size
    ymax = h-grid_size + offset_y
    rows = (h-grid_size)//grid_size
    cropped_img = image[offset_y:ymax, offset_x:xmax]

    unique_tiles = []
    num_tiles = 0
    num_dupes = 0

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
            for r in range(toprow, botrow+1):
                for c in range(leftcol, rightcol+1):
                    ignore_idxs.append((r, c))

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in ignore_idxs:
                num_tiles += 1
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
        out_tiles.append(encoded_tile.tobytes())
    return (out_tiles, offset_y, offset_x, ignore_idxs, unique_tiles)


def unique_tiles_all_offsets(args):
    # file = os.path.join(args.data_path, args.file)
    orig_image = cv2.imread(args.file, cv2.IMREAD_UNCHANGED)
    if orig_image.shape[2] == 4:
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGRA2BGR)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    if args.ui_position == 'top':
        orig_image = orig_image[args.ui_height:, :, :]
    else:
        h = orig_image.shape[0]
        orig_image = orig_image[:h-args.ui_height, :, :]

    #TODO: YOLO detect sprites, ignore boxes
    if args is not None:
        sprited_image, covered_indices, sprite_locs = detect_sprites(
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
    dask.visualize(out, filename='offsets.svg')
    res = dask.compute(out)

    if args.visualize:
        plt.figure()
        a = plt.imshow(sprited_image)

        show_images(res[0][1][4], cols=7)
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
                        default=7, help='num tile sets to select')
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
    print(f'file: {args.file} - num {file_name}')
    pbar = ProgressBar()
    pbar.register()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    # os.makedirs("output", exist_ok=True)
    # image_folder = ImageFolder('data/'+args.game_dir + 'img/')
    # print('image num: ', len(image_folder))
    # Set up model
    model = Darknet(args.model_def, img_size=416).to(device)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()  # Set in evaluation mode
    args.yolo_model = model
    args.classes = load_classes(args.class_path)

    tile_sets = unique_tiles_all_offsets(args)
    res = k_best_sets(tile_sets, args.k)
    pickle.dump(res, open(f'{args.dest}/{args.game}_{file_name}.tiles', 'wb'))

    # print(len(res))
    tset_lens = []
    y_offsets = []
    x_offsets = []
    ignored_tiles = []
    for i in res:
        tset_lens.append(len(i[0]))
        y_offsets.append((i[1]))
        x_offsets.append((i[2]))
        ignored_tiles.append(len(i[3]))
        # print(len(i[0]), i[1], i[2])
    df = pd.DataFrame({"tile_set len": tset_lens,
                       "y_offset": y_offsets, "x_offset": x_offsets, "ignored_tiles": ignored_tiles})
    df.to_csv(f'{args.dest}/{args.game}_{file_name}.csv')
    # for i in range(20):
    #     x = pickle.load(open(f'saved/{i}.tiles', 'rb'))
    #     print(f'file: {i}.tiles')
    #     print(len(x))
    #     for i in x:
    #         print(len(i[0]), i[1], i[2])
