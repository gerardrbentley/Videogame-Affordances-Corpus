import os
import glob
import click
from datetime import datetime
import uuid
import numpy as np
import csv
import json
import pickle

from flask import current_app
from flask import g
from flask.cli import with_appcontext

from sqlalchemy import create_engine, bindparam, String, Integer, DateTime, LargeBinary, Boolean
from sqlalchemy.sql import text
from sqlalchemy.dialects.postgresql import UUID

import image_processing as P
import cv2


def get_image_files(dir=os.path.join('/games')):
    out_files = []
    for game in list_games(dir):
        game_screenshot_files = glob.glob(
            os.path.join(dir, game, 'screenshots', '*.png'))
        game_tile_files = glob.glob(
            os.path.join(dir, game, 'tiles', '*.png'))
        # game_sprite_files = glob.glob(
        #     os.path.join(dir, game, 'sprite', '*.png'))
        out_files.append(
            (game, game_screenshot_files, game_tile_files))

    return out_files


def list_games(dir=os.path.join('/games')):
    try:
        games = next(os.walk(dir))[1]
    except (StopIteration):
        games = []
        print('GAME FOLDERS NOT FOUND IN GAMES, NO FILES FOUND')
    return games


def curr_timestamp():
    return (datetime.now())


def affords_from_csv_file(file, file_name):
    if os.path.isfile(file):
        with open(file, mode='r') as tile_csv:
            csv_reader = csv.DictReader(tile_csv)
            for row in csv_reader:
                if row['file_name'] == file_name:
                    return row
    return None


def offsets_from_csv_file(file, file_uuid):
    if os.path.isfile(file):
        with open(file, mode='r') as offsets_csv:
            csv_reader = csv.DictReader(offsets_csv)
            for row in csv_reader:
                if row['file_name'] == file_uuid:
                    out = (int(row['y_offset']), int(row['x_offset']))
                    return out
    return (0, 0)


def offsets_from_json(screenshots_dir, file_uuid):
    pth = os.path.join(screenshots_dir, file_uuid, f'{file_uuid}.json')
    if os.path.isfile(pth):
        with open(pth, mode='r') as offsets_file:
            data = json.load(offsets_file)
            return (data['y_offset'], data['x_offset'])
    return (0, 0)


def metadata_from_json(screenshots_dir, file_uuid):
    pth = os.path.join(screenshots_dir, file_uuid, f'{file_uuid}.json')
    if os.path.isfile(pth):
        with open(pth, mode='r') as metadata_file:
            data = json.load(metadata_file)
            output = {
                'crop_l': data['crop_l'],
                'crop_r': data['crop_r'],
                'crop_b': data['crop_b'],
                'crop_t': data['crop_t'],
                'ui_x': data['ui_x'],
                'ui_y': data['ui_y'],
                'ui_width': data['ui_width'],
                'ui_height': data['ui_height'],
                'y_offset': data['y_offset'],
                'x_offset': data['x_offset']
            }
            return output
    return {
                'crop_l': 0,
                'crop_r': 0,
                'crop_b': 0,
                'crop_t': 0,
                'ui_x': 0,
                'ui_y': 0,
                'ui_width': 0,
                'ui_height': 0,
                'y_offset': 0,
                'x_offset': 0
            }


def ingest_filesystem_data(dir=os.path.join('/games')):
    total_ingested = {}
    for game, screenshot_files, tile_files in get_image_files(dir):
        # num_images, num_tags = ingest_screenshot_files_with_offsets(
        #         screenshot_files, game, dir)
        num_images, num_tags, num_skipped = ingest_screenshots(
            game, os.path.join(dir, game, 'screenshots'))

        # num_tiles = ingest_tiles_from_pickle(game, dir)
        num_tiles = ingest_tile_files(tile_files, game, dir)
        # ingest_sprite_files(sprite_files, game), dir
        total_ingested[game] = {
                'num_images': num_images, 'num_screenshot_tags': num_tags, 'num_tiles': num_tiles, 'skipped_images': num_skipped}
    print('TOTALS: {}'.format(total_ingested))


def ingest_screenshot_files_with_offsets(files, game, dir):
    offsets_csv = os.path.join(
        dir, game, f'{game}_min_unique_lengths_offsets.csv')

    ctr = 0
    tag_ctr = 0

    for screen_file in files:
        file_name = os.path.split(screen_file)[1]
        screenshot_uuid = os.path.splitext(file_name)[0]
        is_in = check_uuid_in_screenshots(screenshot_uuid)
        if is_in:
            print(f'SKIPPED INGESTING IMAGE: {screenshot_uuid}')
        else:
            y_offset, x_offset = offsets_from_csv_file(
                offsets_csv, screenshot_uuid)
            print('offsets got for image num: {}, y:{}, x:{}'.format(
                screenshot_uuid, y_offset, x_offset))

            cv_image, encoded_png = P.load_image(screen_file)
            h, w, *_ = cv_image.shape
            data = encoded_png.tobytes()

            result = insert_screenshot(
                screenshot_uuid, game, int(w), int(h), y_offset, x_offset, data)

            #TODO: Load known labels from numpy
            labels = P.load_label(
                screen_file)
            if labels is not None:
                ingest_screenshot_tags(labels, screenshot_uuid)
                tag_ctr += 1
            ctr += 1
    return ctr, tag_ctr


def ingest_screenshots(game, screenshots_dir):
    ctr = 0
    tag_ctr = 0
    skip_ctr = 0
    image_folders = next(os.walk(screenshots_dir))[1]

    for screenshot_uuid in image_folders:
        screenshot_file = os.path.join(
            screenshots_dir, screenshot_uuid, f'{screenshot_uuid}.png')
        # screenshot_uuid = os.path.splitext(file_name)[0]
        is_in = check_uuid_in_screenshots(screenshot_uuid)
        if is_in:
            print(f'SKIPPED INGESTING IMAGE: {screenshot_uuid}')
            skip_ctr += 1
        else:
            metadata = metadata_from_json(
                screenshots_dir, screenshot_uuid)
            print('offsets got for image num: {}, y:{}, x:{}'.format(
                screenshot_uuid, metadata['y_offset'], metadata['x_offset']))

            cv_image, encoded_png = P.load_image(screenshot_file)
            h, w, *_ = cv_image.shape
            data = encoded_png.tobytes()

            result = insert_screenshot(
                screenshot_uuid, game, int(w), int(h), data, **metadata)

        #TODO: Load known labels from numpy
        label_files = glob.glob(os.path.join(
            screenshots_dir, screenshot_uuid, "*.npy"))
        if len(label_files) > 0:
            for label_file in label_files:
                tagger_npy = os.path.split(label_file)[1]
                tagger = os.path.splitext(tagger_npy)[0]
                has_tagged = check_tagger_tagged_screenshot(
                    screenshot_uuid, tagger)
                if has_tagged:
                    print(
                        f'SKIPPED INGESTING Tags:{tagger} on {screenshot_uuid}')
                else:
                    label = P.load_label_from_tagger(label_file)
                    if label is not None:
                        ingest_screenshot_tags(
                            label, screenshot_uuid, tagger=tagger)
                        tag_ctr += 1
        ctr += 1
    return ctr, tag_ctr, skip_ctr


def export_to_filesystem(dest='/out_dataset'):
    game_names = get_game_names()
    print(f'exporting data for games: {game_names}')
    total_exported = {}
    for game in game_names:
        screenshot_ctr = 0

        game_path = os.path.join(dest, game['game'])
        os.makedirs(os.path.join(game_path, 'screenshots'), exist_ok=True)
        os.makedirs(os.path.join(game_path, 'tiles'), exist_ok=True)
        os.makedirs(os.path.join(game_path, 'sprites'), exist_ok=True)

        print(f'Made Directories for game: {game}, {game_path}')
        screenshots = get_screenshots_by_game(game['game'])
        print(f'Exporting {len(screenshots)} screenshots for {game}')
        for screenshot in screenshots:
            image_id = screenshot['image_id']
            image_folder = os.path.join(
                game_path, 'screenshots', str(image_id))
            os.makedirs(image_folder, exist_ok=True)

            image_file = os.path.join(image_folder, f'{image_id}.png')
            orig_cv, encoded_img = P.from_data_to_cv(screenshot['data'])
            print(
                f'saving file: {image_file}  -- {orig_cv.shape} {type(orig_cv)}')
            cv2.imwrite(image_file, orig_cv)
            save_labels(image_id, image_folder)
            meta = {'y_offset': screenshot['y_offset'],
                    'x_offset': screenshot['x_offset']}
            with open(os.path.join(
                    image_folder, f'{str(image_id)}.json'), 'w') as file:
                json.dump(meta, file)

        tiles = get_tiles_by_game(game['game'])
        print(f'Exporting {len(tiles)} screenshots for {game}')
        tiles_folder = os.path.join(game_path, 'tiles')
        os.makedirs(tiles_folder, exist_ok=True)
        to_csv = []
        for tile in tiles:
            tile_id = tile['tile_id']

            tile_file = os.path.join(tiles_folder, f'{tile_id}.png')
            orig_cv, encoded_img = P.from_data_to_cv(tile['data'])
            print(
                f'saving file: {tile_file}  -- {orig_cv.shape} {type(orig_cv)}')
            cv2.imwrite(tile_file, orig_cv)

            tile_tag_entries = get_tile_affordances(tile_id)

            for db_entry in tile_tag_entries:
                db_entry['file_name'] = db_entry.pop('tile_id')
                to_csv.append(db_entry)
        with open(os.path.join(tiles_folder, 'tile_affordances.csv'), mode='w') as csv_file:
            fieldnames = ['file_name', "solid", "movable", "destroyable",
                          "dangerous", "gettable", "portal", "usable", "changeable", "ui", "tagger_id"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in to_csv:
                writer.writerow(row)
    return 1


def save_labels(image_uuid, image_folder):
    # label_file = os.path.join(image_folder, 'label', f'{image_uuid}.npy')
    db_entries = get_screenshot_affordances(image_uuid)
    row_ctr = 0
    if len(db_entries) % 9 != 0:
        print('NOT MOD 9 TAG ENTRIES FOR IMAGE: {}'.format(image_uuid))
    print('NUM ROWS OF AFFORDANCES {} FOR IMAGE {}'.format(
        len(db_entries), image_uuid))
    while row_ctr < len(db_entries):
        label_to_convert = []
        tagger_id = db_entries[row_ctr]['tagger_id']
        for affordance in range(9):
            db_entry = db_entries[row_ctr]
            if db_entry['affordance'] != affordance:
                print('AFFORDANCES IN WRONG ORDER')
            tag_cv, encoded = P.from_data_to_cv(db_entry['tags'])
            label_to_convert.append(tag_cv)
            row_ctr += 1
        stacked_array = P.images_to_numpy(label_to_convert)
        pth = os.path.join(image_folder, f'{tagger_id}.npy')
        print(f'NUMPY SAVE: saving file: {pth}')
        np.save(os.path.join(image_folder, f'{tagger_id}.npy'), stacked_array)


def ingest_screenshot_tags(stacked_array, image_id, tagger='ingested'):
    channels_dict = P.numpy_to_images(stacked_array)
    print('INGESTING SCREENSHOT: {}'.format(image_id))
    for i, affordance in enumerate(P.AFFORDANCES):
        encoded_channel = channels_dict[affordance]
        channel_data = encoded_channel.tobytes()
        insert_screenshot_tag(image_id, i, tagger, channel_data)
    pass


# def ingest_tiles_from_pickle(game, dir):
#     #Should be one .tiles file in game directory
#     pickle_file = glob.glob(os.path.join(dir, game, '*.tiles'))
#     ctr = 0
#     if len(pickle_file) > 0:
#         pickle_file = pickle_file[0]
#         print('pickle loc: ', pickle_file)
#         try:
#             unique_game_tiles = pickle.load(open(pickle_file, 'rb'))
#             for tile in unique_game_tiles:
#                 full = cv2.imdecode(tile, cv2.IMREAD_UNCHANGED)
#                 data = tile.tobytes()
#                 h, w, *_ = full.shape
#                 result = insert_tile(game, int(w), int(h), data)
#                 ctr += 1
#         except (OSError, IOError, pickle.UnpicklingError) as e:
#             print('Unpickle error! ')
#     return ctr


def ingest_tile_files(tile_files, game, dir):

    ctr = 0
    tag_ctr = 0
    for tile_file in tile_files:
        file_name = os.path.split(tile_file)[1]
        tile_uuid = os.path.splitext(file_name)[0]
        is_in = check_uuid_in_tiles(tile_uuid)
        if is_in:
            print(f'SKIPPED INGESTING TILE: {tile_uuid}')
        else:
            cv, encoded_png = P.load_image(tile_file)
            h, w, c = cv.shape
            data = encoded_png.tobytes()
            result = insert_tile(tile_uuid, game, int(w), int(h), data)
            # tile_id = result['tile_id']

            # TODO TILE AFFORDANCES
            csv_file = os.path.join(dir, game, 'tiles', 'tile_affordances.csv')
            tile_entry = affords_from_csv_file(csv_file, tile_uuid)
            if tile_entry is not None:
                print('TILE HAD AFFORDS')
                insert_tile_tag(
                    tile_uuid, tile_entry['tagger_id'], int(
                        tile_entry['solid']),
                    int(tile_entry['movable']), int(tile_entry['destroyable']), int(tile_entry['dangerous']), int(tile_entry['gettable']), int(tile_entry['portal']), int(tile_entry['usable']), int(tile_entry['changeable']), int(tile_entry['ui']))
                tag_ctr += 1
            ctr += 1
    return ctr

#
# def ingest_tile_tags(affords, tile_id):
#     tagger = 'ingested'
#     insert_tile_tag(tile_id, tagger, affords[0], affords[1], affords[2],
#                     affords[3], affords[4], affords[5], affords[6], affords[7], affords[8])
#     pass
#

# def ingest_sprite_files(sprite_files, game, dir):
#
#         file.write('Ingesting {} sprites for game: {}\n'.format(
#             len(sprite_files), game))
#     ctr = 0
#     tag_ctr = 0
#     for sprite_file in sprite_files:
#         file_name = os.path.split(sprite_file)[1]
#         file_num_str = os.path.splitext(file_name)[0]
#
#         #4 channel with alpha
#         cv, encoded_png = P.load_sprite(sprite_file)
#         h, w, c = cv.shape
#         data = encoded_png.tobytes()
#         result = insert_sprite(game, int(w), int(h), data)
#         sprite_id = result['sprite_id']
#
#         csv_file = os.path.join(dir, game, 'sprite_affordances.csv')
#         sprite_affords = affords_from_csv_file(csv_file, file_num_str)
#         if sprite_affords is not None:
#             ingest_sprite_tags(sprite_affords, sprite_id)
#             tag_ctr += 1
#         ctr += 1
#
#         file.write('Ingested {} sprites, {} tags for game: {}\n'.format(
#             ctr, tag_ctr, game))
#
#
# def ingest_sprite_tags(affords, sprite_id):
#     tagger = 'ingested'
#     insert_sprite_tag(sprite_id, tagger, affords[0], affords[1], affords[2],
#                       affords[3], affords[4], affords[5], affords[6], affords[7], affords[8])
#     pass
#


def insert_screenshot(image_uuid, game, width, height, image, y_offset=0, x_offset=0, crop_l=0, crop_r=0, crop_t=0, crop_b=0, ui_x=0, ui_y=0, ui_height=0, ui_width=0):
    cmd = text(
        """INSERT INTO screenshots(image_id, game, width, height, y_offset, x_offset, created_on, data, crop_l, crop_r, crop_t, crop_b, ui_x, ui_y, ui_height, ui_width)
        VALUES(:u, :g, :w, :h, :y, :x, :dt, :i, :crop_l, :crop_r, :crop_t, :crop_b, :ui_x, :ui_y, :ui_height, :ui_width)
        RETURNING image_id
        """
    )
    cmd = cmd.bindparams(
        bindparam('u', value=image_uuid, type_=UUID),
        bindparam('g', value=game, type_=String),
        bindparam('w', value=width, type_=Integer),
        bindparam('h', value=height, type_=Integer),
        bindparam('y', value=y_offset, type_=Integer),
        bindparam('x', value=x_offset, type_=Integer),
        bindparam('dt', value=curr_timestamp(), type_=DateTime),
        bindparam('i', value=image, type_=LargeBinary),
        bindparam('crop_l', value=crop_l, type_=Integer),
        bindparam('crop_r', value=crop_r, type_=Integer),
        bindparam('crop_t', value=crop_t, type_=Integer),
        bindparam('crop_b', value=crop_b, type_=Integer),
        bindparam('ui_x', value=ui_x, type_=Integer),
        bindparam('ui_y', value=ui_y, type_=Integer),
        bindparam('ui_height', value=ui_height, type_=Integer),
        bindparam('ui_width', value=ui_width, type_=Integer),
    )
    res = get_connection().execute(cmd)

    for row in res:
        output = {
            'image_id': row['image_id'],
        }
    return output


def insert_screenshot_tag(image_id, affordance, tagger, data):
    cmd = text(
        """INSERT INTO screenshot_tags(image_id, affordance, tagger_id, created_on, tags)
        VALUES(:u, :a, :t, :dt, :d)
        ON CONFLICT ON CONSTRAINT screenshot_tags_pkey
        DO UPDATE SET tags = :d
        """
    )
    cmd = cmd.bindparams(
        bindparam('u', value=image_id, type_=UUID),
        bindparam('a', value=affordance, type_=Integer),
        bindparam('t', value=tagger, type_=String),
        bindparam('dt', value=curr_timestamp(), type_=DateTime),
        bindparam('d', value=data, type_=LargeBinary)
    )
    get_connection().execute(cmd)


def insert_tile(tile_uuid, game, width, height, image):
    cmd = text(
        """INSERT INTO tiles(tile_id, game, width, height, created_on, data)
        VALUES(:u, :g, :w, :h, :dt, :i)
        RETURNING tile_id
        """
    )
    cmd = cmd.bindparams(
        bindparam('u', value=tile_uuid, type_=UUID),
        bindparam('g', value=game, type_=String),
        bindparam('w', value=width, type_=Integer),
        bindparam('h', value=height, type_=Integer),
        bindparam('dt', value=curr_timestamp(), type_=DateTime),
        bindparam('i', value=image, type_=LargeBinary)
    )
    res = get_connection().execute(cmd)

    for row in res:
        output = {
            'tile_id': row['tile_id'],
        }
    return output


def insert_tile_tag(tile_id, tagger, solid, movable, destroyable, dangerous, gettable, portal, usable, changeable, ui):
    cmd = text(
        """INSERT INTO tile_tags(tile_id, created_on, tagger_id, solid, movable, destroyable, dangerous, gettable, portal, usable, changeable, ui)
        VALUES(:ti, :dt, :ta, :s, :m, :de, :da, :g, :p, :us, :c, :ui)
        ON CONFLICT ON CONSTRAINT tile_tags_pkey
        DO UPDATE SET solid = :s, movable = :m, destroyable = :de, dangerous = :da, gettable = :g, portal = :p, usable = :us, changeable = :c, ui = :ui
        """
    )
    cmd = cmd.bindparams(
        bindparam('ti', value=tile_id, type_=UUID),
        bindparam('dt', value=curr_timestamp(), type_=DateTime),
        bindparam('ta', value=tagger, type_=String),
        bindparam('s', value=solid, type_=Boolean),
        bindparam('m', value=movable, type_=Boolean),
        bindparam('de', value=destroyable, type_=Boolean),
        bindparam('da', value=dangerous, type_=Boolean),
        bindparam('g', value=gettable, type_=Boolean),
        bindparam('p', value=portal, type_=Boolean),
        bindparam('us', value=usable, type_=Boolean),
        bindparam('c', value=changeable, type_=Boolean),
        bindparam('ui', value=ui, type_=Boolean),
    )
    get_connection().execute(cmd)


def insert_sprite(sprite_uuid, game, width, height, image):
    cmd = text(
        """INSERT INTO sprites(sprite_id, game, width, height, created_on, data)
        VALUES(:u, :g, :w, :h, :dt, :i)
        RETURNING sprite_id
        """
    )
    cmd = cmd.bindparams(
        bindparam('u', value=sprite_uuid, type_=UUID),
        bindparam('g', value=game, type_=String),
        bindparam('w', value=width, type_=Integer),
        bindparam('h', value=height, type_=Integer),
        bindparam('dt', value=curr_timestamp(), type_=DateTime),
        bindparam('i', value=image, type_=LargeBinary)
    )
    res = get_connection().execute(cmd)

    for row in res:
        output = {
            'sprite_id': row['sprite_id'],
        }
    return output


def insert_sprite_tag(sprite_id, tagger, solid, movable, destroyable, dangerous, gettable, portal, usable, changeable, ui):
    cmd = text(
        """INSERT INTO sprite_tags(sprite_id, created_on, tagger_id, solid, movable, destroyable, dangerous, gettable, portal, usable, changeable, ui)
        VALUES(:ti, :dt, :ta, :s, :m, :de, :da, :g, :p, :us, :c, :ui)
        """
    )
    cmd = cmd.bindparams(
        bindparam('ti', value=sprite_id, type_=UUID),
        bindparam('dt', value=curr_timestamp(), type_=DateTime),
        bindparam('ta', value=tagger, type_=String),
        bindparam('s', value=solid, type_=Boolean),
        bindparam('m', value=movable, type_=Boolean),
        bindparam('de', value=destroyable, type_=Boolean),
        bindparam('da', value=dangerous, type_=Boolean),
        bindparam('g', value=gettable, type_=Boolean),
        bindparam('p', value=portal, type_=Boolean),
        bindparam('us', value=usable, type_=Boolean),
        bindparam('c', value=changeable, type_=Boolean),
        bindparam('ui', value=ui, type_=Boolean),
    )
    get_connection().execute(cmd)


def screenshot_dictify(row):
    output = {
            'image_id': row['image_id'],
            'game': row['game'],
            'width': row['width'],
            'height': row['height'],
            'y_offset': row['y_offset'],
            'x_offset': row['x_offset'],
            'data': row['data'],
            'crop_l': row['crop_l'],
            'crop_r': row['crop_r'],
            'crop_b': row['crop_b'],
            'crop_t': row['crop_t'],
            'ui_x': row['ui_x'],
            'ui_y': row['ui_y'],
            'ui_width': row['ui_width'],
            'ui_height': row['ui_height'],
        }
    return output


def get_random_screenshot():
    cmd = text(
        """SELECT * FROM screenshots
        OFFSET floor(random() * (SELECT COUNT (*) FROM screenshots))
        LIMIT 1;
        """
    )

    res = get_connection().execute(cmd)

    for row in res:
        output = screenshot_dictify(row)
    return output


def get_untagged_screenshot(tagger_id='default'):
    cmd = text(
        """SELECT * FROM screenshots
        WHERE NOT EXISTS
            (SELECT 1
            FROM screenshot_tags
            WHERE screenshots.image_id = screenshot_tags.image_id AND screenshot_tags.tagger_id = :t)
        ORDER BY random()
        LIMIT 1;
        """
    )
    cmd = cmd.bindparams(
        bindparam('t', value=tagger_id, type_=String)
    )

    res = get_connection().execute(cmd)

    for row in res:
        output = screenshot_dictify(row)
    return output


def get_screenshot_by_id(id):
    cmd = text(
        """SELECT * FROM screenshots
        WHERE image_id = :id;
        """
    )
    cmd = cmd.bindparams(
        bindparam('id', value=id, type_=String),
    )
    res = get_connection().execute(cmd)

    for row in res:
        output = screenshot_dictify(row)
    return output


def check_if_already_tagged(image_id, tagger_id):
    cmd = text(
        """SELECT affordance FROM screenshot_tags
        WHERE image_id = :id AND tagger_id = :t;
        """
    )
    cmd = cmd.bindparams(
        bindparam('id', value=image_id, type_=String),
        bindparam('t', value=tagger_id, type_=String)
    )
    res = get_connection().execute(cmd)
    for row in res:
        return True
    return False


def get_screenshot_affordances(id):
    cmd = text(
        """SELECT image_id, affordance, tags, tagger_id FROM screenshot_tags
        WHERE image_id = :id ORDER BY tagger_id, affordance;
        """
    )
    cmd = cmd.bindparams(
        bindparam('id', value=id, type_=String),
    )

    res = get_connection().execute(cmd)
    output = []
    for row in res:
        output.append({
            'image_id': row['image_id'],
            'affordance': row['affordance'],
            'tags': row['tags'],
            'tagger_id': row['tagger_id'],
        })
    return output


def get_game_names():
    cmd = text(
        """SELECT DISTINCT game FROM screenshots;
        """
    )

    res = get_connection().execute(cmd)
    output = []
    for row in res:
        output.append({
            'game': row['game'],
        })
    return output


def get_tile_affordances(tile_id):
    cmd = text(
        """SELECT * FROM tile_tags
        WHERE tile_id = :id;
        """
    )
    cmd = cmd.bindparams(
        bindparam('id', value=tile_id, type_=String),
    )
    res = get_connection().execute(cmd)
    output = []
    for row in res:
        output.append({
            'tile_id': row['tile_id'],
            'tagger_id': row['tagger_id'],
            'solid': int(row['solid']),
            "movable": int(row['movable']),
            "destroyable": int(row['destroyable']),
            "dangerous": int(row['dangerous']),
            "gettable": int(row['gettable']),
            "portal": int(row['portal']),
            "usable": int(row['usable']),
            "changeable": int(row['changeable']),
            "ui": int(row['ui'])
        })
    return output


def get_screenshots_by_game(game):
    cmd = text(
        """SELECT * FROM screenshots
        WHERE game = :g;
        """
    )
    cmd = cmd.bindparams(
        bindparam('g', value=game, type_=String),
    )
    res = get_connection().execute(cmd)

    output = []
    for row in res:
        output.append(screenshot_dictify(row))
    return output


def get_tiles_by_game(game):
    cmd = text(
        """SELECT * FROM tiles
        WHERE game = :g;
        """
    )
    cmd = cmd.bindparams(
        bindparam('g', value=game, type_=String),
    )
    res = get_connection().execute(cmd)

    output = []
    for row in res:
        output.append({
            'tile_id': row['tile_id'],
            'game': row['game'],
            'width': row['width'],
            'height': row['height'],
            'data': row['data'],
        })
    return output


def get_sprites_by_game(game):
    cmd = text(
        """SELECT * FROM sprites
        WHERE game = :g;
        """
    )
    cmd = cmd.bindparams(
        bindparam('g', value=game, type_=String),
    )
    res = get_connection().execute(cmd)

    output = []
    for row in res:
        output.append({
            'sprite_id': row['sprite_id'],
            'game': row['game'],
            'width': row['width'],
            'height': row['height'],
            'data': row['data'],
        })
    return output


def check_uuid_in_screenshots(id):
    cmd = text(
        """SELECT EXISTS(SELECT 1 FROM screenshots where image_id = :i) as "exists"
        """
    )
    cmd = cmd.bindparams(
        bindparam('i', value=id, type_=UUID),
    )
    res = get_connection().execute(cmd)
    for row in res:
        return row['exists']


def check_tagger_tagged_screenshot(id, tagger):
    cmd = text(
        """SELECT EXISTS(SELECT 1 FROM screenshot_tags where image_id = :i and tagger_id = :t) as "exists"
        """
    )
    cmd = cmd.bindparams(
        bindparam('i', value=id, type_=UUID),
        bindparam('t', value=tagger, type_=String),
    )
    res = get_connection().execute(cmd)
    for row in res:
        return row['exists']


def check_uuid_in_tiles(id):
    cmd = text(
        """SELECT EXISTS(SELECT 1 FROM tiles where tile_id = :i) as "exists"
        """
    )
    cmd = cmd.bindparams(
        bindparam('i', value=id, type_=UUID),
    )
    res = get_connection().execute(cmd)
    for row in res:
        return row['exists']


def init_db():
    screenshot_table = text(
        """CREATE TABLE IF NOT EXISTS screenshots(
        image_id UUID PRIMARY KEY,
        game VARCHAR (50) NOT NULL,
        width integer,
        height integer,
        y_offset integer,
        x_offset integer,
        crop_l integer,
        crop_r integer,
        crop_b integer,
        crop_t integer,
        ui_x integer,
        ui_y integer,
        ui_width integer,
        ui_height integer,
        created_on TIMESTAMP NOT NULL,
        data bytea
        )"""
    )
    screenshot_tags_table = text(
        """CREATE TABLE IF NOT EXISTS screenshot_tags(
        image_id UUID NOT NULL,
        affordance integer NOT NULL,
        tagger_id VARCHAR(50) NOT NULL,
        created_on TIMESTAMP NOT NULL,
        tags bytea,
        PRIMARY KEY (image_id, affordance, tagger_id),
        CONSTRAINT screenshot_tags_image_id_fkey FOREIGN KEY (image_id)
          REFERENCES screenshots (image_id) MATCH SIMPLE
          ON UPDATE NO ACTION ON DELETE NO ACTION
        )"""
    )

    tile_table = text(
        """CREATE TABLE IF NOT EXISTS tiles(
        tile_id UUID PRIMARY KEY,
        game VARCHAR (50) NOT NULL,
        width integer,
        height integer,
        created_on TIMESTAMP NOT NULL,
        data bytea
        )"""
    )
    tile_tags_table = text(
        """CREATE TABLE IF NOT EXISTS tile_tags(
        tile_id UUID NOT NULL,
        created_on TIMESTAMP NOT NULL,
        tagger_id VARCHAR(50) NOT NULL,
        solid boolean NOT NULL,
        movable boolean NOT NULL,
        destroyable boolean NOT NULL,
        dangerous boolean NOT NULL,
        gettable boolean NOT NULL,
        portal boolean NOT NULL,
        usable boolean NOT NULL,
        changeable boolean NOT NULL,
        ui boolean NOT NULL,
        PRIMARY KEY (tile_id, tagger_id),
        CONSTRAINT tile_tags_tile_id_fkey FOREIGN KEY (tile_id)
          REFERENCES tiles (tile_id) MATCH SIMPLE
          ON UPDATE NO ACTION ON DELETE NO ACTION
        )"""
    )
    sprite_table = text(
        """CREATE TABLE IF NOT EXISTS sprites(
        sprite_id UUID PRIMARY KEY,
        game VARCHAR (50) NOT NULL,
        width integer,
        height integer,
        created_on TIMESTAMP NOT NULL,
        data bytea
        )"""
    )
    sprite_tag_table = text(
        """CREATE TABLE IF NOT EXISTS sprite_tags(
        sprite_id UUID NOT NULL,
        created_on TIMESTAMP NOT NULL,
        tagger_id VARCHAR(50) NOT NULL,
        solid boolean NOT NULL,
        movable boolean NOT NULL,
        destroyable boolean NOT NULL,
        dangerous boolean NOT NULL,
        gettable boolean NOT NULL,
        portal boolean NOT NULL,
        usable boolean NOT NULL,
        changeable boolean NOT NULL,
        ui boolean NOT NULL,
        PRIMARY KEY (sprite_id, tagger_id),
        CONSTRAINT sprite_tags_sprite_id_fkey FOREIGN KEY (sprite_id)
          REFERENCES sprites (sprite_id) MATCH SIMPLE
          ON UPDATE NO ACTION ON DELETE NO ACTION
        )"""
    )
    to_exec = [screenshot_table, screenshot_tags_table, tile_table,
               tile_tags_table, sprite_table, sprite_tag_table]
    conn = get_connection()
    for cmd in to_exec:
        conn.execute(cmd)


def drop_all():
    tables = ['screenshots', 'screenshot_tags',
              'tiles', 'tile_tags', 'sprites', 'sprite_tags']
    BASE = "DROP TABLE IF EXISTS {} CASCADE"
    conn = get_connection()

    for table in tables:
        cmd = text(BASE.format(table))
        conn.execute(cmd)


def get_connection():
    url = current_app.config['SQLALCHEMY_DATABASE_URI']
    print(url)
    engine = create_engine(url, echo=True)

    return engine


def get_db():
    """Connect to the application's configured database. The connection
    is unique for each request and will be reused if this is called
    again.
    """
    # if "db" not in g:
    #     g.db = sqlite3.connect(
    #         current_app.config["DATABASE"], detect_types=sqlite3.PARSE_DECLTYPES
    #     )
    #     g.db.row_factory = sqlite3.Row

    return None


def close_db(e=None):
    """If this request connected to the database, close the
    connection.
    """
    print('CLOSE DB called')
    pass


@click.command("reset-db")
@with_appcontext
def reset_db_command():
    """Clear existing data and create new tables and ingest pickles and images."""
    drop_all()
    init_db()
    ingest_filesystem_data('../affordances_corpus/tagging_party/')

    click.echo("Initialized the database.")

#@click.command("destroy-db")
#@with_appcontext
#def destroy_db_command():
#    """Clear existing tables."""
#    drop_all()
#    click.echo("Destroyed the database.")


#@click.command("test-db")
#@with_appcontext
#def test_db_command():
#    """test insert."""
#    insert_screenshot('test_game', 256, 224)
#    click.echo("Inserted the database.")


def init_app(app):
   """Register database functions with the Flask app. This is called by
   the application factory.
   """
   app.teardown_appcontext(close_db)
   app.cli.add_command(reset_db_command)
