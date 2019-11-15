from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from flask import current_app
from flask import jsonify
from flask import send_from_directory
from werkzeug.exceptions import abort

import logging
import os
import base64
import json

import db as db
import image_processing as P


# from flaskr.auth import login_required
# from flaskr.db import get_db

logger = logging.getLogger(__name__)
bp = Blueprint("tagger", __name__)

IMAGE_BASE = "data:image/png;base64,{}"
BASE_DIR = os.path.join('..', 'affordances_corpus', 'games')


def b64_string(data):
    return IMAGE_BASE.format((base64.b64encode(data)).decode('utf-8'))


def un_base(input):
    # logger.debug(f'in {type(input)}: {input}')
    remove_format = input[22:]
    # logger.debug(f'no_format {type(remove_format)}: {remove_format}')
    bs = base64.b64decode(remove_format)
    # logger.debug(f'bs {type(bs)}: {bs}')
    to_insert = P.from_bytes_to_grayscale_bytes(bs)

    # cv, _ = P.from_data_to_cv(bs)
    # cv2.imwrite('/app/from_client.png', cv)
    # logger.debug(f'cv {type(cv)}: {cv.shape}')

    # gray, _ = P.from_data_to_cv(to_insert)
    # cv2.imwrite('/app/from_client_gray.png', gray)
    # logger.debug(f'Processed {type(gray)}: {gray.shape}')

    return to_insert


def map_dict(func, dict):
    for k, v in dict.items():
        dict[k] = func(v)


def encode_tile_from_dict(entry):
    entry['tile_data'] = b64_string(entry['tile_data'])
    return entry


@bp.route("/")
def tag_image():
    tagger_id = request.args.get(
        'tagger-id', default='default-tagger', type=str)
    return render_template('base2.html', tagger_id=tagger_id)

@bp.route("/instruct")
def instruct():
    tagger_id = request.args.get(
        'tagger-id', default='default-tagger', type=str)
    return render_template('instruct.html', tagger_id=tagger_id)


@bp.route("/devjs")
def static_js():
    logger.debug('DEV JS CALLED')
    return send_from_directory('templates/js/', 'scripts.js')

@bp.route("/devexamples")
def static_png():
    logger.debug('DEV examlpes CALLED')
    return send_from_directory('templates/', 'examples.png')


@bp.route("/devcss")
def static_css():
    logger.debug('DEV CSS CALLED')
    logger.debug('CSS/Style.cs')
    return send_from_directory('templates/css/', 'style.css')


@bp.route("/devjson")
def test_json():
    with open('templates/example_data.json') as file:
        data = json.load(file)
    logger.debug('DEV JSON CALLED')
    return data


@bp.route("/get_image")
def get_image_to_tag():
    """Return random image, list of unique tiles and locations"""
    tagger_id = request.args.get(
        'tagger-id', default='default-tagger', type=str)
    if tagger_id == 'default-tagger':
        logger.debug("NO TAGGER ID IN GET IMAGE")

    logger.debug('Fetching image for tagger: {}'.format(tagger_id))

    image_data = db.get_untagged_screenshot(tagger_id)
    image_id = image_data['image_id']
    game = image_data['game']
    data = image_data['data']

    meta = {i: image_data[i] for i in image_data if i
            != 'data' and i != 'game' and i != 'image_id'}
    y_offset = image_data['y_offset']
    x_offset = image_data['x_offset']
    logger.debug("Untagged Image data retrieved image_id: {}".format(image_id))
    logger.debug(f'image meta info: {meta}')

    orig_cv, encoded_img = P.from_data_to_cv(data)
    image_string = b64_string(encoded_img)
    logger.debug('Image stringified')

    # known_game_tiles = db.get_tiles_by_game(game)

    unique_tiles = P.unique_tiles_using_meta(
        (orig_cv), **meta)

    tiles_to_tag = get_tile_ids(unique_tiles, game)
    # map_dict(encode_tile_from_dict, tiles_to_tag)
    logger.debug("Unique TILES found: {}".format(len(unique_tiles)))
    logger.debug('Tiles id-d, LEN: {}'.format(len(tiles_to_tag)))

    # tags = P.load_label(image_file)
    # tag_images = P.numpy_to_images(tags)
    # map_dict(b64_string, tag_images)
    output = {
        'image': image_string,
        'image_id': image_id,
        'tiles': tiles_to_tag,
        'y_offset': y_offset,
        'x_offset': x_offset
    }
    logger.debug('base route ok')
    return jsonify({'output': output})


def get_tile_ids(unique_tiles, game):
    known_game_tiles = db.get_tiles_by_game(game)

    tiles_to_tag = {}
    logger.debug('LEN KNOWN TILES: {}'.format(len(known_game_tiles)))
    hit_ctr = 0
    miss_ctr = 0
    for idx, screenshot_tile in enumerate(unique_tiles):
        to_compare = screenshot_tile['tile_data']
        is_in_db = False
        for tile_info in known_game_tiles:
            cv_img, encoded_img = P.from_data_to_cv(tile_info['data'])
            err = P.mse(to_compare, (cv_img))
            if err < 0.001:
                is_in_db = True
                hit_ctr += 1
                # logger.debug("MATCHED {}".format(tile_info['tile_id']))
                # logger.debug("NUM LOCS {}".format(
                #     len(screenshot_tile['locations'])))
                tiles_to_tag['tile_{}'.format(idx)] = {
                    'tile_id': tile_info['tile_id'],
                    'tile_data': b64_string(P.from_cv_to_bytes(to_compare)),
                    'locations': screenshot_tile['locations']
                    }
                break
        if not is_in_db:
            logger.debug("TILE NOT MATCHED IN DB")
            miss_ctr += 1
            tiles_to_tag['tile_{}'.format(idx)] = {
                'tile_id': -1,
                'tile_data': b64_string(P.from_cv_to_bytes(to_compare)),
                'locations': screenshot_tile['locations']
                }
        # idx = 0
        # if idx == -1:
        #     logger.debug('NEW TILE FOUND')
        #     height, width, channels = screenshot_tile['tile_data'].shape
        #     tile_data = P.from_cv_to_bytes(screenshot_tile['tile_data'])
        #     db.insert_tile(game, width, height, tile_data)
    logger.debug(f'db tile hits: {hit_ctr}, misses: {miss_ctr}')
    return tiles_to_tag


@bp.route("/submit_tags", methods=['POST'])
def save_affordances():
    """Save affordances for a certain tile"""
    data = request.get_json(force=True)
    tagger = data['tagger_id']
    image_id = data['image_id']
    logger.debug(f'RECEIVED TAGS FROM: {tagger} FOR IMAGE: {image_id}')
    # logger.debug(f'{data}')
    tiles = data['tiles']
    insert_count = 0
    skip_count = 0
    for tile in tiles:
        tile_id = tiles[tile]['tile_id']
        if not isinstance(tile_id, int):
            logger.debug('DB INSERT TILE TAGS ID: {}'.format(
                tile_id))

            db.insert_tile_tag(tiles[tile]['tile_id'], tagger, tiles[tile]['solid'], tiles[tile]['movable'],
                               tiles[tile]['destroyable'], tiles[tile]['dangerous'], tiles[tile]['gettable'], tiles[tile]['portal'], tiles[tile]['usable'], tiles[tile]['changeable'], tiles[tile]['ui'])
            insert_count += 1
        else:
            skip_count += 1

    logger.debug('INSERTED {} Tile Tags. SKIPPED {} Tiles. SUBMITTED: {}'.format(
        insert_count, skip_count, len(tiles)))

    tag_images = data['tag_images']
    affordance_count = 0
    for affordance in tag_images:
        data = tag_images[affordance]
        to_insert = un_base(data)
        affordance_num = P.AFFORDANCES.index(affordance)
        logger.debug('DB INSERT IMAGE TAGS for afford: {} {}, data type: {}'.format(
            affordance, affordance_num, type(to_insert)))
        # data = convert_img_to_encoding(tag_img['data'])
        db.insert_screenshot_tag(image_id, affordance_num, tagger, to_insert)
    logger.debug(f'num affordance channels: {len(tag_images)}')

    output = {
        'Success': True,
        'Response': 200,
    }

    return output


@bp.route("/testinsert")
def test_insert():
    db.insert_screenshot('test_game', 256, 224)
    return 'INSERTED'


@bp.route('/dev')
def homepage():
    tagger_id = 'test_tagger'
    # image_data = db.get_untagged_screenshot(tagger_id)
    image_data = db.get_screenshot_by_id(
        '670fecdf-02c2-47ad-becd-20b6ddac5fc0')
    image_id = image_data['image_id']

    game = image_data['game']
    y_offset = image_data['y_offset']
    x_offset = image_data['x_offset']
    # width = image_data['width']
    # height = image_data['height']
    data = image_data['data']
    logger.debug("Untagged Image data retrieved image_id: {}".format(image_id))

    orig_cv, encoded_img = P.from_data_to_cv(data)
    image_string = b64_string(encoded_img)
    logger.debug('Image stringified')

    # known_game_tiles = db.get_tiles_by_game(game)
    # logger.debug('Known tiles loaded')

    output = {
        'tagger_id': tagger_id,
        'image': image_string,
        'image_id': image_id,
    }
    tags = db.get_screenshot_affordances(image_id)
    if len(tags) % 9 != 0:
        logger.debug(f'WRONG NUM OF AFFORDANCES FOR IMAGE: {image_id}')
        return render_template('homepage.html', **output)

    to_convert = []
    for affordance in range(9):
        db_entry = tags[affordance + 9]
        if db_entry['affordance'] % 9 != affordance:
            logger.debug(
                f'AFFORDANCE TAG WRONG ORDER {affordance}, for image: {image_id}')
            return render_template('homepage.html', **output)
        orig_bw, encoded_tag = P.from_data_to_cv(db_entry['tags'])
        logger.debug(f'test db tags: {orig_bw.shape}, {type(orig_bw)}')
        output[P.AFFORDANCES[affordance]] = b64_string(encoded_tag)
        to_convert.append(orig_bw)

    stacked_array = P.images_to_numpy(to_convert)
    logger.debug(
        f'got tag array: {stacked_array.shape}, {type(stacked_array)}, {stacked_array.min()}, {stacked_array.max()}')
    # if len(to_convert) != 9:
    #     logger.debug(f'NOT ALL AFFORDANCES FOR IMAGE: {image_id}')
    #     return 0

    # map_dict(b64_string, to_convert)
    logger.debug('{}'.format(output.keys()))
    return render_template('homepage.html', **output)


"""
DEPRECATED
"""
"""
@bp.route("/get_image")
def get_image_to_tag():
    #TODO: get tagger_id from cookie or POST
    tagger_id = 'developer'

    logger.debug('Fetching image for tagger: {}'.format(tagger_id))
    image_data = db.get_untagged_screenshot(tagger_id)
    image_id = image_data['image_id']

    game = image_data['game']
    # width = image_data['width']
    # height = image_data['height']
    data = image_data['data']
    logger.debug("Untagged Image data retrieved image_id: {}".format(image_id))

    orig_cv, encoded_img = P.from_data_to_cv(data)
    image_string = b64_string(encoded_img)
    logger.debug('Image stringified')

    tiles_to_tag = {}
    unique_tiles = P.find_unique_tiles(orig_cv, game)
    logger.debug("LEN UNIQUE TILES: {}".format(len(unique_tiles)))
    tiles_to_tag = get_tile_ids(
        unique_tiles, game)
    logger.debug("LEN TILES: {}".format(len(tiles_to_tag)))

    logger.debug('Tiles stringified')

    # tags = P.load_label(image_file)
    # tag_images = P.numpy_to_images(tags)
    # map_dict(b64_string, tag_images)
    output = {
        'image': image_string,
        'image_id': image_id,
        'tiles': tiles_to_tag,
    }
    logger.debug('base route ok')
    return jsonify({'output': output})


def get_tile_ids(unique_tiles, game):
    known_game_tiles = db.get_tiles_by_game(game)

    tiles_to_tag = {}
    logger.debug('LEN KNOWN TILES: {}'.format(len(known_game_tiles)))
    idx = 0
    for screenshot_tile in unique_tiles:
        to_compare = screenshot_tile['tile_data']
        for tile_info in known_game_tiles:
            cv_img, encoded_img = P.from_data_to_cv(tile_info['data'])
            err = P.mse(to_compare, cv_img)
            if err < 0.001:
                logger.debug("MATCHED {}".format(tile_info['tile_id']))
                logger.debug("DATA {}".format(b64_string(encoded_img)))
                logger.debug("LOCS {}".format(screenshot_tile['locations']))
                tiles_to_tag['tile_{}'.format(idx)] = {
                    'tile_id': tile_info['tile_id'],
                    'tile_data': b64_string(encoded_img),
                    'locations': screenshot_tile['locations']
                    }
                idx += 1
                break
        # idx = 0
        # if idx == -1:
        #     logger.debug('NEW TILE FOUND')
        #     height, width, channels = screenshot_tile['tile_data'].shape
        #     tile_data = P.from_cv_to_bytes(screenshot_tile['tile_data'])
        #     db.insert_tile(game, width, height, tile_data)
    return tiles_to_tag
"""
