import os
import cv2
import numpy as np

AFFORDANCES = ["solid", "movable", "destroyable",
               "dangerous", "gettable", "portal", "usable", "changeable", "ui"]
DEFAULTS = {'loz': {'ui_height': 56, 'grid_size': 16, 'ui_position': 'top'},
            'sm3': {'ui_height': 40, 'grid_size': 8, 'ui_position': 'bot'},
            'metroid': {'ui_height': 0, 'grid_size': 16, 'ui_position': 'top'}
            }


def from_cv_to_bytes(cv_img):
    orig, encoded = cv2.imencode('.png', cv_img)
    data = encoded.tobytes()
    return data


def from_data_to_cv(db_data):
    data = db_data.tobytes()
    encoded_img = np.frombuffer(data, dtype=np.uint8)
    orig_cv = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
    return orig_cv, encoded_img


def load_image(image_file=os.path.join('..', 'affordances_corpus', 'games', 'loz', 'img', '0.png')):
    orig_cv = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    channels = cv2.split(orig_cv)
    if(len(channels) == 1):
        orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_GRAY2BGR)
    if orig_cv.shape[2] == 4:
        orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_BGRA2BGR)
    # orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2RGB)
    _, image = cv2.imencode('.png', orig_cv)
    return orig_cv, image


def load_sprite(sprite_file=os.path.join('..', 'affordances_corpus', 'games', 'loz', 'sprite', '1.png')):
    orig_cv = cv2.imread(sprite_file, cv2.IMREAD_UNCHANGED)
    channels = cv2.split(orig_cv)
    if(len(channels) == 1):
        orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_GRAY2BGRA)
        channels = cv2.split(orig_cv)
    if(len(channels) == 3):
        orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2BGRA)
        channels = cv2.split(orig_cv)
    _, image = cv2.imencode('.png', orig_cv)
    return orig_cv, image


def load_label(image_file=os.path.join('..', 'affordances_corpus', 'games', 'loz', 'img', '0.png')):
    label_file = image_file.replace('img', 'label').replace('png', 'npy')
    if os.path.isfile(label_file):
        # print('Label File Found')
        stacked_array = np.load(label_file)
    else:
        # print('New Label File')
        stacked_array = None
    return stacked_array


def numpy_to_images(arr):
    _, _, channels = arr.shape
    output = {}
    for i in range(channels):
        one_channel = arr[:, :, i].copy() * 255
        _, image_buffer = cv2.imencode('.png', one_channel)
        output[AFFORDANCES[i]] = image_buffer
    return output


def mse(a, b):
    if a.shape != b.shape:
        return 1
    diffs = np.square(np.subtract(a, b))
    total_diff = np.sum(diffs)
    return np.divide(total_diff, (a.shape[0] * a.shape[1]))


def location_in_list(new_tile, prev_tiles):
    for i in range(len(prev_tiles)):
        old_tile = prev_tiles[i]
        err = mse(new_tile, old_tile)
        # print('SSIM: {:.2f}'.format(similarity))
        if err < 0.001:
            return i
    return -1


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
Returns list of dictionaries opencv tiles with corresponding list of locations in image
"""


def find_unique_tiles(image, game):
    print('Finding unique tiles in img')
    settings = DEFAULTS[game]
    grid_size = settings['grid_size']
    ui_position = settings['ui_position']
    ui_height = settings['ui_height']
    grid_offset_x = 0
    grid_offset_y = 0

    height, width, channels = image.shape
    img_tiles = []
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

                img_tiles.append({
                    'tile_data': template_np,
                    'locations': matches_dict
                    })
                tile_ctr += 1
            else:
                skip_ctr += 1

    print('VISITED {} tiles, sum of unique({}) + skip({}) = {}'.format(
        len(visited_locations), len(img_tiles), skip_ctr, (len(img_tiles)+skip_ctr)))
    return img_tiles