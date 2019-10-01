
import cv2
import numpy as np
import dask
from dask.diagnostics import ProgressBar

import os
import argparse
import glob


def mse(a, b):
    diffs = np.square(np.subtract(a, b))
    total_diff = np.sum(diffs)
    return np.divide(total_diff, (a.shape[0] * a.shape[1]))


def is_unique_by_mse(new_tile, prev_tiles):
    for old_tile in prev_tiles:
        err = mse(new_tile, old_tile)
        # print('SSIM: {:.2f}'.format(similarity))
        if err < 0.001:
            return False

    return True


@dask.delayed
def get_unique_tiles(image, offset_y=0, offset_x=0, grid_size=16):
    h, w, c = image.shape
    cropped_img = np.copy(
        image)[offset_y:h-grid_size+offset_y, offset_x:w-grid_size+offset_x]
    rows = (h-grid_size)//grid_size
    cols = (w-grid_size)//grid_size
    unique_tiles = []
    num_tiles = 0
    num_dupes = 0
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

    return (unique_tiles, offset_y, offset_x)


@dask.delayed
def unique_tiles_all_offsets(args, file):
    orig_image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if orig_image.shape[2] == 4:
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGRA2BGR)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    if args.ui_position == 'top':
        orig_image = orig_image[args.ui_height:, :, :]
    else:
        h = orig_image.shape[0]
        orig_image = orig_image[:h-args.ui_height, :, :]

    # print('orig image shape: {}'.format(orig_image.shape))
    unique_tiles_per_offset = {}
    num_tiles_per_offset = {}

    potential_offsets = [(y, x) for x in range(0, args.grid_size)
                         for y in range(0, args.grid_size)]

    out = [get_unique_tiles(orig_image, y, x, grid_size=args.grid_size)
           for (y, x) in potential_offsets]
    # for (y, x) in potential_offsets:
    #     unique_tile_tuple = get_unique_tiles(
    #         orig_image, y, x, grid_size=args.grid_size)

    # sorted_dict = [(k, num_tiles_per_offset[k])
    #                for k in sorted(num_tiles_per_offset, key=num_tiles_per_offset.get)]
    # for i, (k, v) in enumerate(sorted_dict):
    #     if i >= num:
    #         break
    #     print('Rank {} offset: {}, unique tiles: {}'.format(
    #         i+1, k, v))
    #     out.append((k, unique_tiles_per_offset[k]))
    return out


def len_unique_tiles(input):
    return len(input[0])


@dask.delayed
def min_unique_tiles(offset_tile_lists):
    best_min = 10000
    best_idx = 0
    for i in range(len(offset_tile_lists)):
        curr = offset_tile_lists[i]
        curr_len = len_unique_tiles(curr)
        if curr_len < best_min:
            best_min = curr_len
            best_idx = i
    return offset_tile_lists[best_idx]


def best_set(k_offset_list):
    output = []
    for k_offsets in k_offset_list:
        output.append(min_unique_tiles(k_offsets))

    return output


def predict_all_offsets(args):
    game_image_files = glob.glob(os.path.join(
        args.data_path, args.game_dir, 'img', '*.png'))
    print(len(game_image_files))
    # results = []
    # for file in game_image_files:
    #     print('Offsets for file: {}'.format(file))
    #     results.append(unique_tiles_all_offsets(args, file))
    results = [unique_tiles_all_offsets(args, file)
               for file in game_image_files]
    out = dask.delayed(best_set)(results)
    out.visualize(filename='offsets.svg')
    end = out.compute()
    for i in range(len(end)):
        print(end[i].compute())


def parse_args():
    parser = argparse.ArgumentParser(description='grid detection')

    parser.add_argument('--data-path', type=str, default='../../affordances_corpus/games',
                        help='path to games folder')
    parser.add_argument('--game-dir', type=str, default='loz',
                        help='game directory name')
    parser.add_argument('--file-num', type=int, default=1,
                        help='image num to tag')
    parser.add_argument('--grid-size', type=int,
                        default=16, help='grid square size')
    parser.add_argument('--ui-height', type=int,
                        default=56, help='ignore this range')
    parser.add_argument('--ui-position', type=str,
                        default='top', help='ui top or bot')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    pbar = ProgressBar()
    pbar.register()
    predict_all_offsets(args)
