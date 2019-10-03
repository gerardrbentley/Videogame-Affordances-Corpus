import argparse
import os
import glob
import pickle

import cv2
import numpy as np


def load_tiles(dir):
    templates = {}

    for file in glob.glob(os.path.join(dir, '*.png')):
        file_name = os.path.split(file)[1]
        name = os.path.splitext(file_name)[0]
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        channels = cv2.split(img)
        if(len(channels) == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        templates[int(name)] = img

    return templates


def map_decode(img):
    out = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    return out


def get_best_offset(matches, grid_size):
    mod_x = np.zeros((grid_size-1,), dtype=np.uint8)
    mod_y = np.zeros((grid_size-1,), dtype=np.uint8)

    for (y, x) in matches:
        out_x = x % grid_size
        out_y = y % grid_size
        mod_x[out_x] += 1
        mod_y[out_y] += 1

    print(mod_x, mod_x.argmax())
    print(mod_y, mod_y.argmax())
    return (mod_y.argmax(), mod_x.argmax())


def main(args):
    base_image = cv2.imread(args.file, cv2.IMREAD_UNCHANGED)
    if base_image.shape[2] == 4:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGRA2BGR)
    known_tiles = pickle.load(open('unique_set_sm3_406.tiles', 'rb'))
    known_tiles = list(map(map_decode, known_tiles))
    # known_tiles = load_tiles(args.tiles_dir)
    print('image shape: ', base_image.shape)
    height, width, channels = base_image.shape

    recorded_matches = {}
    all_matches = []
    for tile_num, tile in enumerate(known_tiles):
        res = cv2.matchTemplate(
            tile, base_image, cv2.TM_SQDIFF_NORMED)
        loc = np.where(res <= 5e-6)
        matches = list(zip(*loc[::1]))
        if len(matches) != 0:
            recorded_matches[tile_num] = matches
            all_matches.extend(matches)

    potential_offsets = [(y, x) for x in range(0, args.grid_size)
                         for y in range(0, args.grid_size)]
    matches_per_offset = {}
    for tile_num, matches in recorded_matches.items():
        print(tile_num, len(matches), matches)

    offset_y, offset_x = get_best_offset(all_matches, args.grid_size)


def parse_args():
    parser = argparse.ArgumentParser(description='grid detection')

    parser.add_argument('--tiles-dir', type=str, default='../../affordances_corpus/tagging_party/sm3/tile_img',
                        help='path to tiles folder')
    parser.add_argument('--img-dir', type=str, default='../../affordances_corpus/tagging_party/sm3/img',
                        help='path to img folder')
    parser.add_argument('--file', type=str, default='../../affordances_corpus/tagging_party/sm3/img/1.png',
                        help='path to img folder')
    parser.add_argument('--game', type=str, default='sm3',
                        help='game directory name')

    parser.add_argument('--grid-size', type=int,
                        default=16, help='grid square size')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
