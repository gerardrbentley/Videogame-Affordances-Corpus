import os
import glob
import click
from datetime import datetime
import base64
import numpy as np
import csv
import pickle

from flask import current_app
from flask import g
from flask.cli import with_appcontext

from sqlalchemy import create_engine, bindparam, String, Integer, DateTime, LargeBinary, Boolean
from sqlalchemy.sql import text

import image_processing as P
import cv2


''' Lists all png images with file structure DIR/GAME_NAME/img/0.png'''


def get_image_files(dir=os.path.join('games')):
    out_files = []
    for game in list_games(dir):
        per_game_files = glob.glob(os.path.join(dir, game, 'img', '*.png'))
        # game_tile_files = glob.glob(
        #     os.path.join(dir, game, 'tile_img', '*.png'))
        # game_sprite_files = glob.glob(
        #     os.path.join(dir, game, 'sprite', '*.png'))
        out_files.append(
            (game, per_game_files))

    return out_files


def list_games(dir=os.path.join('games')):
    games = next(os.walk(dir))[1]
    return games


def curr_timestamp():
    return (datetime.now())


def affords_from_csv_file(file, file_num_str):
    if os.path.isfile(file):
        with open(file, mode='r') as tile_csv:
            csv_reader = csv.DictReader(tile_csv)
            for row in csv_reader:
                if row['file'] == file_num_str:
                    out = []
                    for x in row:
                        if x != 'file':
                            out.append(bool(int(row[x])))
                    return out
    return None


def offsets_from_csv_file(file, file_num_str):
    if os.path.isfile(file):
        with open(file, mode='r') as offsets_csv:
            csv_reader = csv.DictReader(offsets_csv)
            for row in csv_reader:
                if row['file_num'] == file_num_str:
                    out = (int(row['y_offset']), int(row['x_offset']))
                    return out
    return (0, 0)


def ingest_filesystem_data(dir=os.path.join('games')):
    total_ingested = {}
    for game, screenshot_files in get_image_files(dir):
        num_images = ingest_screenshot_files_with_offsets(
            screenshot_files, game, dir)
        num_tiles = ingest_tiles_from_pickle(game, dir)
        # ingest_tile_files(tile_files, game, dir)
        # ingest_sprite_files(sprite_files, game), dir
        total_ingested[game] = {
            'num_images': num_images, 'num_tiles': num_tiles}
    print('TOTALS: {}'.format(total_ingested))


def ingest_screenshot_files_with_offsets(files, game, dir):
    offsets_csv = os.path.join(
        dir, game, f'{game}_min_unique_lengths_offsets.csv')

    ctr = 0
    tag_ctr = 0

    for screen_file in files:
        file_name = os.path.split(screen_file)[1]
        file_num_str = os.path.splitext(file_name)[0]
        y_offset, x_offset = offsets_from_csv_file(
            offsets_csv, file_num_str)
        print('offsets got for image num: {}, y:{}, x:{}'.format(
            file_num_str, y_offset, x_offset))

        cv_image, encoded_png = P.load_image(screen_file)
        h, w, *_ = cv_image.shape
        data = encoded_png.tobytes()

        result = insert_screenshot(
            game, int(w), int(h), y_offset, x_offset, data)

        #TODO: Load known labels from numpy
        # image_id = result['image_id']
        # label = P.load_label(screen_file)
        # if label is not None:
        #     ingest_screenshot_tags(label, image_id)
        #     tag_ctr += 1
        ctr += 1
    return ctr


#
# def ingest_screenshot_tags(stacked_array, image_id):
#     channels_dict = P.numpy_to_images(stacked_array)
#     tagger = 'ingested'
#     for i, affordance in enumerate(P.AFFORDANCES):
#         encoded_channel = channels_dict[affordance]
#         channel_data = encoded_channel.tobytes()
#         insert_screenshot_tag(image_id, i, tagger, channel_data)
#     pass


def ingest_tiles_from_pickle(game, dir):
    #Should be one .tiles file in game directory
    pickle_file = glob.glob(os.path.join(dir, game, '*.tiles'))
    ctr = 0
    if len(pickle_file) > 0:
        pickle_file = pickle_file[0]
        print('pickle loc: ', pickle_file)
        unique_game_tiles = pickle.load(open(pickle_file, 'rb'))
        for tile in unique_game_tiles:
            full = cv2.imdecode(tile, cv2.IMREAD_UNCHANGED)
            data = tile.tobytes()
            h, w, *_ = full.shape
            result = insert_tile(game, int(w), int(h), data)
            ctr += 1
    return ctr

#
# def ingest_tile_files(tile_files, game, dir):
#
#         file.write('Ingesting {} tiles for game: {}\n'.format(
#             len(tile_files), game))
#     ctr = 0
#     tag_ctr = 0
#     for tile_file in tile_files:
#         file_name = os.path.split(tile_file)[1]
#         file_num_str = os.path.splitext(file_name)[0]
#
#         cv, encoded_png = P.load_image(tile_file)
#         h, w, c = cv.shape
#         data = encoded_png.tobytes()
#         result = insert_tile(game, int(w), int(h), data)
#         tile_id = result['tile_id']
#
#         csv_file = os.path.join(dir, game, 'tile_affordances.csv')
#         tile_affords = affords_from_csv_file(csv_file, file_num_str)
#         if tile_affords is not None:
#             ingest_tile_tags(tile_affords, tile_id)
#             tag_ctr += 1
#         ctr += 1
#
#         file.write('Ingested {} tiles, {} tags for game: {}\n'.format(
#             ctr, tag_ctr, game))

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


def insert_screenshot(game, width, height, y_offset, x_offset, image):
    cmd = text(
        """INSERT INTO screenshots(image_id, game, width, height, y_offset, x_offset, created_on, data)
        VALUES(DEFAULT, :g, :w, :h, :y, :x, :dt, :i)
        RETURNING image_id
        """
    )
    cmd = cmd.bindparams(
        bindparam('g', value=game, type_=String),
        bindparam('w', value=width, type_=Integer),
        bindparam('h', value=height, type_=Integer),
        bindparam('y', value=y_offset, type_=Integer),
        bindparam('x', value=x_offset, type_=Integer),
        bindparam('dt', value=curr_timestamp(), type_=DateTime),
        bindparam('i', value=image, type_=LargeBinary)
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
        VALUES(:i, :a, :t, :dt, :d)
        """
    )
    cmd = cmd.bindparams(
        bindparam('i', value=image_id, type_=Integer),
        bindparam('a', value=affordance, type_=Integer),
        bindparam('t', value=tagger, type_=String),
        bindparam('dt', value=curr_timestamp(), type_=DateTime),
        bindparam('d', value=data, type_=LargeBinary)
    )
    get_connection().execute(cmd)


def insert_tile(game, width, height, image):
    cmd = text(
        """INSERT INTO tiles(tile_id, game, width, height, created_on, data)
        VALUES(DEFAULT, :g, :w, :h, :dt, :i)
        RETURNING tile_id
        """
    )
    cmd = cmd.bindparams(
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
        """
    )
    cmd = cmd.bindparams(
        bindparam('ti', value=tile_id, type_=Integer),
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


def insert_sprite(game, width, height, image):
    cmd = text(
        """INSERT INTO sprites(sprite_id, game, width, height, created_on, data)
        VALUES(DEFAULT, :g, :w, :h, :dt, :i)
        RETURNING sprite_id
        """
    )
    cmd = cmd.bindparams(
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
        bindparam('ti', value=sprite_id, type_=Integer),
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


def get_random_screenshot():
    cmd = text(
        """SELECT * FROM screenshots
        OFFSET floor(random() * (SELECT COUNT (*) FROM screenshots))
        LIMIT 1;
        """
    )

    res = get_connection().execute(cmd)

    for row in res:
        output = {
            'image_id': row['image_id'],
            'game': row['game'],
            'width': row['width'],
            'height': row['height'],
            'y_offset': row['y_offset'],
            'x_offset': row['x_offset'],
            'data': row['data'],
        }
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
        output = {
            'image_id': row['image_id'],
            'game': row['game'],
            'width': row['width'],
            'height': row['height'],
            'y_offset': row['y_offset'],
            'x_offset': row['x_offset'],
            'data': row['data'],
        }
    return output


def get_screenshot_by_id(id):
    cmd = text(
        """SELECT * FROM screenshots
        WHERE image_id = :id;
        """
    )
    cmd = cmd.bindparams(
        bindparam('id', value=id, type_=Integer),
    )
    res = get_connection().execute(cmd)

    for row in res:
        output = {
            'image_id': row['image_id'],
            'game': row['game'],
            'width': row['width'],
            'height': row['height'],
            'y_offset': row['y_offset'],
            'x_offset': row['x_offset'],
            'data': row['data'],
        }
    return output


def check_if_already_tagged(image_id, tagger_id):
    cmd = text(
        """SELECT affordance FROM screenshot_tags
        WHERE image_id = :id AND tagger_id = :t;
        """
    )
    cmd = cmd.bindparams(
        bindparam('id', value=image_id, type_=Integer),
        bindparam('t', value=tagger_id, type_=String)
    )
    res = get_connection().execute(cmd)
    for row in res:
        return True
    return False


def get_screenshot_affordances(id):
    cmd = text(
        """SELECT image_id, affordance, tags, tagger_id FROM screenshot_tags
        WHERE image_id = :id;
        """
    )
    cmd = cmd.bindparams(
        bindparam('id', value=id, type_=Integer),
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


def get_tile_affordances(tile_id):
    cmd = text(
        """SELECT * FROM tile_tags
        WHERE tile_id = :id;
        """
    )
    cmd = cmd.bindparams(
        bindparam('id', value=tile_id, type_=Integer),
    )
    res = get_connection().execute(cmd)

    for row in res:
        output = {
            'tile_id': row['tile_id'],
            'solid': row['solid'],
            "movable": row['movable'],
            "destroyable": row['destroyable'],
            "dangerous": row['dangerous'],
            "gettable": row['gettable'],
            "portal": row['portal'],
            "usable": row['usable'],
            "changeable": row['changeable'],
            "ui": row['ui']
        }
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


def init_db():
    screenshot_table = text(
        """CREATE TABLE IF NOT EXISTS screenshots(
        image_id serial PRIMARY KEY,
        game VARCHAR (50) NOT NULL,
        width integer,
        height integer,
        y_offset integer,
        x_offset integer,
        created_on TIMESTAMP NOT NULL,
        data bytea
        )"""
    )
    screenshot_tags_table = text(
        """CREATE TABLE IF NOT EXISTS screenshot_tags(
        image_id integer NOT NULL,
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
        tile_id serial PRIMARY KEY,
        game VARCHAR (50) NOT NULL,
        width integer,
        height integer,
        created_on TIMESTAMP NOT NULL,
        data bytea
        )"""
    )
    tile_tags_table = text(
        """CREATE TABLE IF NOT EXISTS tile_tags(
        tile_id integer NOT NULL,
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
        sprite_id serial PRIMARY KEY,
        game VARCHAR (50) NOT NULL,
        width integer,
        height integer,
        created_on TIMESTAMP NOT NULL,
        data bytea
        )"""
    )
    sprite_tag_table = text(
        """CREATE TABLE IF NOT EXISTS sprite_tags(
        sprite_id integer NOT NULL,
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
