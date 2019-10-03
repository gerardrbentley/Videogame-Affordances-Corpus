
import cv2
import numpy as np
import dask
from dask.diagnostics import ProgressBar
import pandas as pd

import os
import argparse
import pickle


def mse(a, b):
    diffs = np.square(np.subtract(a, b))
    total_diff = np.sum(diffs)
    return np.divide(total_diff, (a.shape[0] * a.shape[1]))


def is_unique_by_mse(new_tile, prev_tiles):
    for old_tile in prev_tiles:
        err = mse(new_tile, old_tile)
        # print('SSIM: {:.2f}'.format(similarity))
        if err < 0.00001:
            return False

    return True


@dask.delayed
def get_unique_tiles(image, offset_y=0, offset_x=0, grid_size=16):
    h, w, c = image.shape
    cropped_img = image[offset_y:h-grid_size
                        + offset_y, offset_x:w-grid_size+offset_x]
    rows = (h-grid_size)//grid_size
    cols = (w-grid_size)//grid_size
    unique_tiles = []
    num_tiles = 0
    num_dupes = 0

    #TODO: YOLO detect sprites, ignore boxes

    for r in range(rows):
        for c in range(cols):
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
    return (out_tiles, offset_y, offset_x)


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

    potential_offsets = [(y, x) for x in range(0, args.grid_size)
                         for y in range(0, args.grid_size)]
    out = []
    for (y, x) in potential_offsets:
        out.append(get_unique_tiles(np.copy(orig_image),
                                    y, x, grid_size=args.grid_size))
    dask.visualize(out, filename='offsets.svg')
    res = dask.compute(out)
    return res[0]


def len_unique_tiles(input):
    return len(input[0])


def k_best_sets(tile_sets_with_offsets, k):
    out_sort = sorted(tile_sets_with_offsets,
                      key=lambda x: len_unique_tiles(x))
    return out_sort[:k]


def parse_args():
    parser = argparse.ArgumentParser(description='grid detection')

    parser.add_argument('--file', type=str, default='./0.png')
    parser.add_argument('--game', type=str, default='sm3')
    parser.add_argument('--k', type=int,
                        default=7, help='num tile sets to select')
    parser.add_argument('--grid-size', type=int,
                        default=16, help='grid square size')
    parser.add_argument('--ui-height', type=int,
                        default=40, help='ignore this range')
    parser.add_argument('--ui-position', type=str,
                        default='bot', help='ui top or bot')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    file_name = os.path.split(args.file)[1]
    file_name = os.path.splitext(file_name)[0]
    print(f'file: {args.file} - num {file_name}')
    pbar = ProgressBar()
    pbar.register()
    tile_sets = unique_tiles_all_offsets(args)
    res = k_best_sets(tile_sets, args.k)
    pickle.dump(res, open(f'saved/{args.game}_{file_name}.tiles', 'wb'))

    # print(len(res))
    tset_lens = []
    y_offsets = []
    x_offsets = []
    for i in res:
        tset_lens.append(len(i[0]))
        y_offsets.append((i[1]))
        x_offsets.append((i[2]))
        # print(len(i[0]), i[1], i[2])
    df = pd.DataFrame({"tile_set len": tset_lens,
                       "y_offset": y_offsets, "x_offset": x_offsets})
    df.to_csv(f'saved/{args.game}_{file_name}.csv')
    # for i in range(20):
    #     x = pickle.load(open(f'saved/{i}.tiles', 'rb'))
    #     print(f'file: {i}.tiles')
    #     print(len(x))
    #     for i in x:
    #         print(len(i[0]), i[1], i[2])
