
import os
import glob
import pickle
import random
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

random.seed(47)


def np_help(x):
    return np.frombuffer(x, dtype=np.uint8)


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


def unique_concat(previous, new):
    out = previous.copy()
    dupes = 0
    for potential in new:
        is_new = True
        for seen in previous:
            if np.array_equal(potential, seen):
                # print('ARR EQUAL')
                is_new = False
                dupes += 1
                break
        if is_new:
            # print('add new')
            # print(type(out), len(out))
            out.append(potential)
            # print(type(out), len(out))
    return out, dupes


def main(args):
    random.seed(47)

    PICKLE_DIR = args.pickle_dir
    CURR_GAME = args.game

    print('Loading Pickles from : {}'.format(PICKLE_DIR))
    tile_sets = {}
    for file in glob.glob(os.path.join(PICKLE_DIR, f'{CURR_GAME}_*.tiles')):
        curr_pickle = pickle.load(open(file, 'rb'))

        file_name = os.path.split(file)[1]
        file_name = os.path.splitext(file_name)[0][4:]
        file_name = int(file_name)
        options = {}
        for i, (tiles, y_offset, x_offset) in enumerate(curr_pickle):
            #TODO convert tiles with np.frombuffer(__, dtype=np.uint8)
            np_tiles = list(map(np_help, tiles))
            options[i] = {'tiles': np_tiles,
                          'y_offset': y_offset, 'x_offset': x_offset}
        tile_sets[file_name] = options
    print(f'Loaded {len(tile_sets)} pickled tile_set options')

    #individuals have length equal to num images
    IND_SIZE = len(tile_sets)

    greedy_set = []

    files = []
    tset_lens = []
    y_offsets = []
    x_offsets = []
    bincount = [0, 0, 0, 0, 0]
    for i in range(IND_SIZE):
        curr_options = tile_sets[i]
        best_next = len(greedy_set) + 100
        best_idx = -1
        best_len = 0
        updated_greedy = []
        for idx in range(5):
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
        bincount[best_idx] += 1
        files.append(i)
        tset_lens.append(best_len)
        y_offsets.append(curr_options[best_idx]['y_offset'])
        x_offsets.append(curr_options[best_idx]['x_offset'])
        greedy_set = updated_greedy
    print('bins: ', bincount)

    # print(len(i[0]), i[1], i[2])
    df = pd.DataFrame({"file_num": files, "tile_set len": tset_lens,
                       "y_offset": y_offsets, "x_offset": x_offsets})
    df.to_csv(f'{CURR_GAME}_min_unique_lengths_offsets.csv', index=False)
    return bincount, greedy_set


def map_decode(img):
    out = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description='decides min unique tile set when local img sets are saved as pickles')

    parser.add_argument('--pickle-dir', type=str, default='saved',
                        help='path to tiles folder')
    parser.add_argument('--game', type=str, default='sm3',
                        help='game name')
    parser.add_argument('--grid-size', type=int,
                        default=8, help='grid square size')
    parser.add_argument('--save-img', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    b, g = main(args)
    print('done')
    length = len(g)
    pickle.dump(g, open(f'unique_set_{args.game}_{length}.tiles', 'wb'))
    ready = list(map(map_decode, g))

    if args.save_img:
        IMGS_PER_GRID = 400
        i = 0
        print(len(ready))
        while i < len(ready):
            to_show = ready[i:i+IMGS_PER_GRID]

            fig = show_images(to_show, cols=20)

            plt.savefig(f'loz_{i}.png')
            # plt.pause(0.0001)
            i += IMGS_PER_GRID
    # plt.show()
