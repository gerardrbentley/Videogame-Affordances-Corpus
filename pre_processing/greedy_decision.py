
import os
import glob
import pickle
import argparse
import uuid
import json

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from pre_process_utils import show_images, unique_concat


def main(args):
    """
    Parameters
    ----------
    args: Must include
        pickle-dir: directory of `{CURR_GAME}_{FILE_NUM}.tiles` files
        game: name of current game

    Saves
    ---------
    {CURR_GAME}_min_unique_lengths_offsets.csv
        records offset chosen for each image file and length of tile set for that image and offset
    {CURR_GAME}_unique_set_{NUM_TILES_FOR_GAME}.tiles
        pickled list of unique cv2 encoded pngs (1d ndarray) for game

    Returns
    ---------
    (bincounts, greedy_set) where
        bincounts: list counting which rank unique tile set was chosen for each file
            (default list of 5 integers, first index should be greatest, as those were min local sets)
        greedy_set: List of unique cv2 encoded pngs (1d ndarray) from combining all pickled sets
    """

    PICKLE_DIR = args.pickle_dir
    CURR_GAME = args.game

    print('Loading Pickled tile sets from : {}'.format(
        os.path.join(PICKLE_DIR, CURR_GAME)))
    tile_sets = {}
    for file in glob.glob(os.path.join(PICKLE_DIR, CURR_GAME, f'{CURR_GAME}_*.tiles')):
        curr_pickle = pickle.load(open(file, 'rb'))

        # Cut path
        file_name = os.path.split(file)[1]
        # Cut string after 'CURR_GAME_'
        file_name = file_name.split('_')[1]
        # Cut .tiles
        file_name = os.path.splitext(file_name)[0]

        options = {}
        for i, (tiles, y_offset, x_offset, _) in enumerate(curr_pickle):
            # np_tiles = list(map(np_help, tiles))
            # tiles is list of cv2 encoded pngs, 1d ndarray
            options[i] = {'tiles': tiles,
                          'y_offset': y_offset, 'x_offset': x_offset}
        # Each options includes best k (default 5 in prediction) tile sets for that file
        tile_sets[file_name] = options
    print(f'Loaded {len(tile_sets)} pickled tile_set options')

    NUM_FILES = len(tile_sets)

    greedy_set = []

    files = []
    tset_lens = []
    y_offsets = []
    x_offsets = []
    bincount = []
    for i in range(args.k):
        bincount.append(0)
    i = 0
    for file_name, curr_options in tile_sets.items():

        # Upper bound for next decision based on accumulated tiles
        best_next = len(greedy_set) + 100
        best_idx = -1
        best_len = 0
        updated_greedy = []

        # Find which tile set for given file adds fewest new tiles
        for idx in range(args.k):
            tile_set = curr_options[idx]['tiles']
            uniques, dupes = unique_concat(greedy_set, tile_set)
            if len(uniques) < best_next:
                # print('better, ', len(tile_set), '..', idx)
                best_idx = idx
                updated_greedy = uniques
                best_next = len(uniques)
                best_len = len(tile_set)
        #FOR CHECKING NON (0,0) OFFSETS
        # if curr_options[best_idx]['y_offset'] != 0 or curr_options[best_idx]['x_offset'] != 0:
        #     print(curr_options[best_idx]['y_offset'],
        #           curr_options[best_idx]['x_offset'], 'file', i)
        #     if SAVE_ERRORS:
        #         z = curr_options[best_idx]['y_offset']
        #         q = curr_options[best_idx]['x_offset']
        #         to_save = list(
        #             map(map_decode, curr_options[best_idx]['tiles']))
        #         show_images(to_save, cols=10)
        #         plt.savefig(f'err_{i}_{z}_{q}.png')
        print(
            f'iteration {i}, prev: {len(greedy_set)}, new greed: {len(updated_greedy)}, diff: {len(updated_greedy) - len(greedy_set)}')
        i += 1
        bincount[best_idx] += 1
        files.append(file_name)
        tset_lens.append(best_len)
        y_offsets.append(curr_options[best_idx]['y_offset'])
        x_offsets.append(curr_options[best_idx]['x_offset'])

        meta = {'y_offset': curr_options[best_idx]['y_offset'],
                'x_offset': curr_options[best_idx]['x_offset']}
        with open(os.path.join(
                args.games_dir, args.game, 'screenshots', file_name, f'{file_name}.json'), 'w') as file:
            json.dump(meta, file)

        greedy_set = updated_greedy
    print('bins: ', bincount)

    df = pd.DataFrame({"file_name": files, "tile_set_len": tset_lens,
                       "y_offset": y_offsets, "x_offset": x_offsets})
    df.to_csv(f'{CURR_GAME}_min_unique_lengths_offsets.csv', index=False)

    length = len(greedy_set)
    os.makedirs(f'{args.pickle_dir}/{args.game}/tiles', exist_ok=True)
    for tile in greedy_set:
        full_tile = cv2.imdecode(tile, cv2.IMREAD_UNCHANGED)
        new_id = str(uuid.uuid4())
        # print('saving: ', new_id)
        cv2.imwrite(
            f'{args.pickle_dir}/{args.game}/tiles/{new_id}.png', full_tile)

    # pickle.dump(greedy_set, open(
    #     f'{args.game}_unique_set_{length}.tiles', 'wb'))
    return bincount, greedy_set


def map_decode(img):
    out = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def parse_args():
    parser = argparse.ArgumentParser(
        description='decides min unique tile set when local img sets are saved as pickles')

    parser.add_argument('--pickle-dir', type=str, default='output',
                        help='path to tiles folder')
    parser.add_argument('--game', type=str, default='sm3',
                        help='game name')
    parser.add_argument('--games-dir', type=str, default='../games')
    parser.add_argument('--grid-size', type=int,
                        default=8, help='grid square size')
    parser.add_argument('--k', type=int,
                        default=5, help='num potential tile sets per image')
    parser.add_argument('--save-img', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    bincounts, greedy_set = main(args)
    print('done')

    if args.save_img:
        os.makedirs('tile_set_imgs', exist_ok=True)
        ready = list(map(map_decode, greedy_set))
        # ready = greedy_set
        IMGS_PER_GRID = 400
        i = 0
        print(len(ready))
        while i < len(ready):
            to_show = ready[i:i+IMGS_PER_GRID]

            fig = show_images(to_show, cols=20)

            plt.savefig(f'tile_set_imgs/{args.game}_{i}.png')
            # plt.pause(0.0001)
            i += IMGS_PER_GRID
    # plt.show()
