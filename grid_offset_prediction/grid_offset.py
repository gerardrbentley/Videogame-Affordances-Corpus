
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import argparse
import glob


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
        plt.imshow(image)
        a.set_title(title)
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    # plt.show()


def is_unique_by_mse(new_tile, prev_tiles):
    for old_tile in prev_tiles:
        err = mse(new_tile, old_tile)
        # print('SSIM: {:.2f}'.format(similarity))
        if err < 0.001:
            return False

    return True

#
# def is_unique_in_list(new_tile, prev_tiles):
#     for old_tile in prev_tiles:
#         similarity = ssim(new_tile, old_tile, multichannel=True)
#         # print('SSIM: {:.2f}'.format(similarity))
#         if similarity > 0.98:
#             return False
#         if similarity > 0.9:
#             pass
#             # print('Sim > 0.9: {:.2f}'.format(similarity))
#     return True


def get_unique_tiles(image, offset_y=0, offset_x=0, prev_best=10000, grid_size=16):
    h, w, c = image.shape
    # print(offset_y, h-grid_size+offset_y, offset_x, w-grid_size+offset_x)
    cropped_img = np.copy(
        image)[offset_y:h-grid_size+offset_y, offset_x:w-grid_size+offset_x]
    rows = (h-grid_size)//grid_size
    cols = (w-grid_size)//grid_size
    # print('Testing, grid divisions: {} rows, {} cols. offset ({}, {})'.format(
    #     rows, cols, offset_y, offset_x))
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

            if len(unique_tiles) > prev_best:
                print('EARLY FAIL {} unique / {} dupe tiles == {} / {} total'.format(len(unique_tiles),
                                                                                     num_dupes, len(unique_tiles)+num_dupes, num_tiles))
                return unique_tiles, cropped_img
    # print('{} unique / {} dupe tiles == {} / {} total'.format(len(unique_tiles),
                # num_dupes, len(unique_tiles)+num_dupes, num_tiles))
    return unique_tiles, cropped_img


def predict_all_offsets(args):
    game_image_files = glob.glob(os.path.join(
        args.data_path, args.game_dir, 'img', '*.png'))
    print(len(game_image_files))
    for file in game_image_files:
        print('Offsets for file: {}'.format(file))
        results = best_k_offsets(args, file, 5)


def best_k_offsets(args, file, num):
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
    min_tile_num = 10000
    min_tile_offset = (-1, -1)
    min_tile_set = None
    min_tile_image = None

    potential_offsets = [(y, x) for x in range(0, args.grid_size)
                         for y in range(0, args.grid_size)]
    for (y, x) in potential_offsets:
        unique_tiles, cropped_img = get_unique_tiles(
            orig_image, y, x, grid_size=args.grid_size, prev_best=min_tile_num)

        num_tiles_per_offset['({},{})'.format(y, x)] = len(unique_tiles)
        unique_tiles_per_offset['({},{})'.format(y, x)] = (unique_tiles)

    sorted_dict = [(k, num_tiles_per_offset[k])
                   for k in sorted(num_tiles_per_offset, key=num_tiles_per_offset.get)]
    for i, (k, v) in enumerate(sorted_dict):
        if i >= num:
            break
        print('Rank {} offset: {}, unique tiles: {}'.format(
            i+1, k, v))
    return 1


def predict_offset(args):
    image_path = os.path.join(
        args.data_path, args.game_dir, 'img', str(args.file_num)+'.png')
    orig_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if orig_image.shape[2] == 4:
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGRA2BGR)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    if args.ui_position == 'top':
        orig_image = orig_image[args.ui_height:, :, :]
    else:
        h = orig_image.shape[0]
        orig_image = orig_image[:h-args.ui_height, :, :]

    print('orig image shape: {}'.format(orig_image.shape))
    tiles_per_offset = {}
    min_tile_num = 10000
    min_tile_offset = (-1, -1)
    min_tile_set = None
    min_tile_image = None

    potential_offsets = [(y, x) for x in range(0, args.grid_size)
                         for y in range(0, args.grid_size)]
    for (y, x) in potential_offsets:
        unique_tiles, cropped_img = get_unique_tiles(
            orig_image, y, x, prev_best=min_tile_num)
        if len(unique_tiles) < min_tile_num:
            print('Old min: {}, New min: {}'.format(
                min_tile_num, len(unique_tiles)))
            print('Old off: {}, New off: {}'.format(min_tile_offset, (y, x)))
            min_tile_num = len(unique_tiles)
            min_tile_offset = (y, x)
            min_tile_set = unique_tiles
            min_tile_image = cropped_img
    print('BEST offset: {}, unique tiles: {}'.format(
        min_tile_offset, min_tile_num))
    #Add grid to orig:
    y = args.grid_size + min_tile_offset[0]
    x = args.grid_size + min_tile_offset[1]
    while x < orig_image.shape[1]:
        orig_image = cv2.line(
            orig_image, (x, 0), (x, orig_image.shape[0]), color=(255, 0, 0), thickness=1)
        x += args.grid_size

    while y < orig_image.shape[0]:
        orig_image = cv2.line(
            orig_image, (0, y), (orig_image.shape[1], y), color=(255, 0, 0), thickness=1)
        y += args.grid_size

    show_images(min_tile_set, cols=8)
    plt.figure()
    plt.title('Cropped and offset {}'.format(min_tile_offset))
    plt.imshow(orig_image)
    plt.pause(0.001)
    plt.show()
    return min_tile_offset


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
    predict_all_offsets(args)
