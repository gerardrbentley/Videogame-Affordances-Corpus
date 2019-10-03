from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from flask import current_app
from flask import jsonify
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


def map_dict(func, dict):
    for k, v in dict.items():
        dict[k] = func(v)


def encode_tile_from_dict(entry):
    entry['tile_data'] = b64_string(entry['tile_data'])
    return entry


@bp.route("/json")
def test_json():
    with open('templates/example_data.json') as file:
        data = json.load(file)
    return data


@bp.route("/testinsert")
def test_insert():
    db.insert_screenshot('test_game', 256, 224)
    return 'INSERTED'


@bp.route('/home')
def homepage():
    tagger_id = 'test_tagger'
    return render_template('homepage.html', tagger_id=tagger_id)


@bp.route("/get_image")
def get_image_to_tag():
    """Return random image, list of unique tiles and locations"""

    #TODO: get tagger_id from cookie or POST
    tagger_id = 'developer'

    logger.debug('Fetching image for tagger: {}'.format(tagger_id))
    image_data = db.get_screenshot_by_id(5)
    #TODO: random image
    # image_data = db.get_untagged_screenshot(tagger_id)
    image_id = image_data['image_id']

    game = image_data['game']
    # width = image_data['width']
    # height = image_data['height']
    data = image_data['data']
    logger.debug("Untagged Image data retrieved image_id: {}".format(image_id))

    orig_cv, encoded_img = P.from_data_to_cv(data)
    image_string = b64_string(encoded_img)
    logger.debug('Image stringified')

    known_game_tiles = db.get_tiles_by_game(game)
    logger.debug('Known tiles loaded')

    tiles_to_tag = P.match_known_tiles(orig_cv, known_game_tiles, game)
    logger.debug("LEN TILES to tag: {}".format(len(tiles_to_tag)))

    map_dict(encode_tile_from_dict, tiles_to_tag)
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


@bp.route("/submit_tags", methods=['POST'])
def save_affordances():
    """Save affordances for a certain tile"""
    data = request.get_json(force=True)
    tagger = data['tagger_id']
    image_id = data['image_id']

    tiles = data['tiles']
    for tile in tiles:
        tile_id = tiles[tile]['tile_id']
        print('DB INSERT TILE TAGS for id: {}'.format(tile_id))

        # db.insert_tile_tag(tile['tile_id'], tagger, tile['solid'], tile['movable'],
        #                    tile['destroyable'], tile['dangerous'], tile['gettable'], tile['portal'], tile['usable'], tile['changeable'], tile['ui'])

    tag_images = data['tag_images']
    for affordance in tag_images:
        print('DB INSERT IMAGE TAGS for afford:', affordance)
        data = tag_images[affordance]
        # data = convert_img_to_encoding(tag_img['data'])
        # db.insert_screenshot_tag(image_id, afford, tagger, data)

    output = {
        'Success': True,
        'Response': 200,
    }

    return output


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
