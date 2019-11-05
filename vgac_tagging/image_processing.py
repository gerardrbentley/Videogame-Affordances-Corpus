import os
import cv2
import numpy as np

AFFORDANCES = ["solid", "movable", "destroyable",
               "dangerous", "gettable", "portal", "usable", "changeable", "ui"]
DEFAULTS = {'loz': {'ui_height': 56, 'grid_size': 16, 'ui_position': 'top'},
            'sm3': {'ui_height': 40, 'grid_size': 8, 'ui_position': 'bot'},
            'metroid': {'ui_height': 0, 'grid_size': 16, 'ui_position': 'top'},
            'donkey_kong': {'ui_height': 0, 'grid_size': 16, 'ui_position': 'top'},
            'gradius': {'ui_height': 0, 'grid_size': 16, 'ui_position': 'top'},
            'guardian_legend': {'ui_height': 0, 'grid_size': 16, 'ui_position': 'top'},
            }


def cv_convert(cv_img):
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)


def from_cv_to_bytes(cv_img):
    orig, encoded = cv2.imencode('.png', cv_img)
    data = encoded.tobytes()
    return data


def from_data_to_cv(db_data, force_grayscale=False):
    if not isinstance(db_data, bytes):
        data = db_data.tobytes()
    else:
        data = db_data
    encoded_img = np.frombuffer(data, dtype=np.uint8)
    orig_cv = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
    if force_grayscale:
        orig_cv = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
    return orig_cv, encoded_img


def from_bytes_to_grayscale_bytes(bs):
    encoded_img = np.frombuffer(bs, dtype=np.uint8)
    gray_cv = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
    return from_cv_to_bytes(gray_cv)


def load_image(image_file=os.path.join('..', 'affordances_corpus', 'games', 'loz', 'img', '0.png')):
    orig_cv = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    channels = cv2.split(orig_cv)
    if(len(channels) == 1):
        orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_GRAY2BGR)
    if orig_cv.shape[2] == 4:
        orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_BGRA2BGR)
    # orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2RGB)
    _, encoded_png = cv2.imencode('.png', orig_cv)
    return orig_cv, encoded_png


def load_sprite(sprite_file=os.path.join('..', 'affordances_corpus', 'games', 'loz', 'sprite', '1.png')):
    orig_cv = cv2.imread(sprite_file, cv2.IMREAD_UNCHANGED)
    channels = cv2.split(orig_cv)
    if(len(channels) == 1):
        orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_GRAY2BGRA)
        channels = cv2.split(orig_cv)
    if(len(channels) == 3):
        orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2BGRA)
        channels = cv2.split(orig_cv)
    orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_BGRA2RGBA)
    _, image = cv2.imencode('.png', orig_cv)
    return orig_cv, image


def load_label(image_file=os.path.join('..', 'affordances_corpus', 'games', 'loz', 'img', '0.png')):
    label_file = image_file.replace(
        'screenshots', 'labels').replace('png', 'npy')
    if os.path.isfile(label_file):
        print('Label File Found {}'.format(label_file))
        stacked_array = np.load(label_file)
    else:
        print('New Label File for {}'.format(label_file))
        stacked_array = None
    return stacked_array


def load_label_from_tagger(label_file):
    if os.path.isfile(label_file):
        print('Label File Found {}'.format(label_file))
        stacked_array = np.load(label_file)
    else:
        print('New Label File for {}'.format(label_file))
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


def images_to_numpy(affordance_images):
    height, width, *_ = affordance_images[0].shape
    output = np.zeros((height, width, 9))
    print(
        f'GEN numpy array of dims {output.shape} for aff_image {affordance_images[0].shape}')
    for i in range(9):
        one_channel = affordance_images[i].copy()
        output[:, :, i] = one_channel // 255
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


def find_unique_tiles(image, game, y_offset, x_offset):
    print('Finding unique tiles in img')
    settings = DEFAULTS[game]
    grid_size = settings['grid_size']
    ui_position = settings['ui_position']
    ui_height = settings['ui_height']
    height, width, channels = image.shape
    img_tiles = []
    visited_locations = []
    tile_ctr = 0
    skip_ctr = 0
    rows, cols = gen_grid(width, height, grid_size, ui_height,
                          ui_position, x_offset, y_offset)
    for r in np.unique(rows):
        for c in np.unique(cols):
            if((r, c) not in visited_locations):
                template_np = image[r:r+grid_size if r+grid_size <= height else height,
                                    c:c+grid_size if c+grid_size <= width else width].copy()
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

#
# def get_best_offset(matches, grid_size):
#     mod_x = np.zeros((grid_size,), dtype=np.uint8)
#     mod_y = np.zeros((grid_size,), dtype=np.uint8)
#
#     for (y, x) in matches:
#         out_x = x % grid_size
#         out_y = y % grid_size
#         mod_x[out_x] += 1
#         mod_y[out_y] += 1
#
#     print(mod_y, mod_y.argmax())
#     print(mod_x, mod_x.argmax())
#     return (mod_y.argmax(), mod_x.argmax())
#
#
# def match_known_tiles(image, known_tiles, game='sm3'):
#     height, width, channels = image.shape
#     settings = DEFAULTS[game]
#     grid_size = settings['grid_size']
#     ui_position = settings['ui_position']
#     ui_height = settings['ui_height']
#     total_matches = []
#     temp = {}
#     tile_ctr = 0
#     #Template match all known tiles
#     for idx, tile_info in enumerate(known_tiles):
#         cv_tile, encoded_img = from_data_to_cv(tile_info['data'])
#         # if idx == 5:
#         #     plt.plot()
#         #     x = plt.imshow(image)
#         #     plt.pause(0.0001)
#         #     plt.show()
#         #     y = plt.imshow(cv_tile)
#         #     plt.pause(0.0001)
#         #     plt.show()
#         #     print(type(cv_tile), cv_tile.shape)
#         #     print(type(encoded_img), encoded_img.shape)
#         #     print(type(image), image.shape)
#         res = cv2.matchTemplate(image, cv_tile, cv2.TM_SQDIFF_NORMED)
#         loc = np.where(res <= 5e-7)
#         matches = list(zip(*loc[::1]))
#
#         if len(matches) != 0:
#             # print('MATCHES')
#             total_matches.extend(matches)
#             id = tile_info['tile_id']
#             temp[f'tile_{tile_ctr}'] = {
#                 'tile_id': id,
#                 'tile_data': encoded_img,
#                 'locations': matches}
#             tile_ctr += 1
#     # print(temp)
#     #Find most matched on grid
#     y_offset, x_offset = get_best_offset(total_matches, grid_size)
#     rows, cols = gen_grid(width, height, grid_size, ui_height,
#                           ui_position, x_offset, y_offset)
#     #elimnate matches not on predicted grid
#     out = {}
#     for key, entry in temp.items():
#         curr_matches = entry['locations']
#         # curr_matches = [(y, x) for (y, x) in curr_matches if point_on_grid(
#         #     x, y, cols, rows)]
#
#         matches_dict = {}
#         for i in range(len(curr_matches)):
#             y, x = curr_matches[i]
#             matches_dict['location_{}'.format(i)] = {
#                                               'x': int(x), 'y': int(y)}
#         if len(curr_matches) != 0:
#             out[key] = {
#                 'tile_id': entry['tile_id'],
#                 'tile_data': entry['tile_data'],
#                 'locations': matches_dict
#             }
#     return out
#
