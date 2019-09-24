import os
import cv2
import numpy as np

AFFORDANCES = ["solid", "movable", "destroyable",
               "dangerous", "gettable", "portal", "usable", "changeable", "ui"]


def load_image(image_file=os.path.join('..', 'affordances_corpus', 'games', 'loz', 'img', '0.png')):
    orig_cv = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    if orig_cv.shape[2] == 4:
        orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_BGRA2BGR)
    # orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2RGB)
    _, image = cv2.imencode('.png', orig_cv)
    return orig_cv, image


def load_tile(image_file=os.path.join('..', 'affordances_corpus', 'games', 'loz', 'tile_img', '0.png')):
    orig_cv = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    if orig_cv.shape[2] == 4:
        orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_BGRA2BGR)
    # orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2RGB)
    _, image = cv2.imencode('.png', orig_cv)
    return orig_cv, image


def load_label(image_file=os.path.join('..', 'affordances_corpus', 'games', 'loz', 'img', '0.png')):
    label_file = image_file.replace('img', 'label').replace('png', 'npy')
    if os.path.isfile(label_file):
        print('Label File Found')
        stacked_array = np.load(label_file)
    else:
        print('New Label File')
        stacked_array = np.full(
            [224, 256, 9], fill_value=0.5)
    return stacked_array


def numpy_to_images(arr):
    _, _, channels = arr.shape
    output = {}
    for i in range(channels):
        one_channel = arr[:, :, i].copy() * 255
        _, image_buffer = cv2.imencode('.png', one_channel)
        output[AFFORDANCES[i]] = image_buffer
    return output


def gen_grid(width, height, grid_size=16, ui_height=0, ui_position='top', grid_offset_x=0, grid_offset_y=0):
    if ui_position == 'top':
        ignore_start = ui_height
    else:
        ignore_start = 0
    row_num = (height - ui_height) // grid_size
    if (row_num * grid_size) + ignore_start + grid_offset_y >= height:
        row_num -= 1
    col_num = (width) // grid_size
    if (col_num * grid_size) + grid_offset_x > width:
        col_num -= 1

    rows, cols = np.indices((row_num, col_num))
    rows = rows * grid_size
    cols = cols * grid_size

    rows = rows + ignore_start + grid_offset_y
    cols = cols + grid_offset_x

    return rows, cols


def point_on_grid(c, r, cols, rows):
    return c in cols and r in rows


"""
Takes opencv image
Returns dictionary of opencv tiles with corresponding list of locations in image
"""


def find_unique_tiles(image, grid_size=16, ui_position='top', ui_height=0, grid_offset_x=0, grid_offset_y=0):
    print('Finding unique tiles in img')
    height, width, channels = image.shape
    img_tiles = {}
    visited_locations = []
    tile_ctr = 0
    skip_ctr = 0
    rows, cols = gen_grid(width, height, grid_size, ui_height,
                          ui_position, grid_offset_x, grid_offset_y)
    for r in np.unique(rows):
        for c in np.unique(cols):
            if((r, c) not in visited_locations):
                template_np = image[r:r+grid_size if r+grid_size <= height else height,
                                    c:c+grid_size if c+grid_size <= width else width].copy()
                _, template_image = cv2.imencode('.png', template_np)
                res = cv2.matchTemplate(
                    template_np, image, cv2.TM_SQDIFF_NORMED)
                loc = np.where(res <= 5e-6)
                matches = list(zip(*loc[::1]))
                matches = [(y, x) for (y, x) in matches if point_on_grid(
                    x, y, cols, rows)]

                matches_dict = {}
                for i in range(len(matches)):
                    y, x = matches[i]
                    matches_dict['location_{}'.format(i)] = {
                                                      'x': int(x), 'y': int(y)}
                if len(matches) != 0:
                    for match_loc in matches:
                        visited_locations.append(match_loc)
                else:
                    print(
                        'ERROR MATCHING TILE WITHIN IMAGE: (r,c) ({},{})'.format(r, c))

                img_tiles['tile_{}'.format(tile_ctr)] = {
                    'tile_data': template_image,
                    'locations': matches_dict
                    }
                tile_ctr += 1
            else:
                skip_ctr += 1

    print('VISITED {} tiles, sum of unique({}) + skip({}) = {}'.format(
        len(visited_locations), len(img_tiles), skip_ctr, (len(img_tiles)+skip_ctr)))
    return img_tiles
