import os
import glob
import click
from datetime import datetime
import base64
import numpy as np

from flask import current_app
from flask import g
from flask.cli import with_appcontext

from sqlalchemy import create_engine, bindparam, String, Integer, DateTime, LargeBinary
from sqlalchemy.sql import text

import image_processing as P

POSTGRES_URL = 'localhost'
POSTGRES_USER = 'gbkh2015'
POSTGRES_PASS = 'dev'
POSTGRES_DB = 'affordances_db'

DB_URL = 'postgresql+psycopg2://{}:{}@{}/{}'.format(
    POSTGRES_USER, POSTGRES_PASS, POSTGRES_URL, POSTGRES_DB)

''' Lists all png images with file structure DIR/GAME_NAME/img/0.png'''


def get_image_files(dir=os.path.join('..', 'affordances_corpus', 'games')):
    image_files = []
    for game in list_games(dir):
        per_game_files = glob.glob(os.path.join(dir, game, 'img', '*.png'))
        image_files.append((game, per_game_files))
    return image_files


def list_games(dir=os.path.join('..', 'affordances_corpus', 'games')):
    games = next(os.walk(dir))[1]
    return games


def curr_timestamp():
    return (datetime.now())


def ingest_screenshots(dir=os.path.join('..', '..', 'affordances_corpus', 'games')):
    for game, file_names in get_image_files(dir):
        for image_file in file_names:
            print('loading f: ', image_file, ' FROM ', game)
            cv, data = P.load_image(image_file)
            h, w, c = cv.shape
            print('w: ', w, 'h: ', h)
            data = data.tobytes()
            print(type(data))
            print(cv.dtype)
            insert_screenshot(game, int(w), int(h), data, dev=DB_URL)
            print('INSERTED')
            return None


def insert_screenshot(game, width, height, image, dev=None):
    cmd = text(
        """INSERT INTO screenshots(image_id, game, width, height, created_on, data)
        VALUES(DEFAULT, :g, :w, :h, :dt, :i)
        """
    )
    cmd = cmd.bindparams(
        bindparam('g', value=game, type_=String),
        bindparam('w', value=width, type_=Integer),
        bindparam('h', value=height, type_=Integer),
        bindparam('dt', value=curr_timestamp(), type_=DateTime),
        bindparam('i', value=image, type_=LargeBinary)
    )
    get_connection(dev).execute(cmd)


def get_random_screenshot(dev=None):
    cmd = text(
        """SELECT * FROM screenshots OFFSET
        floor(random() * (SELECT COUNT (*) FROM screenshots))
        LIMIT 1;
        """
    )
    res = get_connection(dev).execute(cmd)

    for row in res:
        output = {
            'image_id': row['image_id'],
            'game': row['game'],
            'width': row['width'],
            'height': row['height'],
            'data': row['data'],
        }
    return output


def get_screenshot_by_id(id, dev=None):
    cmd = text(
        """SELECT * FROM screenshots
        WHERE image_id = :id
        """
    )
    cmd = cmd.bindparams(
        bindparam('id', value=id, type_=Integer),
    )
    res = get_connection(dev).execute(cmd)

    for row in res:
        output = {
            'image_id': row['image_id'],
            'game': row['game'],
            'width': row['width'],
            'height': row['height'],
            'data': row['data'],
        }
    return output


def init_db():
    screenshot_table = text(
        """CREATE TABLE IF NOT EXISTS screenshots(
        image_id serial PRIMARY KEY,
        game VARCHAR (50) NOT NULL,
        width integer,
        height integer,
        created_on TIMESTAMP NOT NULL,
        data bytea
        )"""
    )
    screenshot_tags_table = text(
        """CREATE TABLE IF NOT EXISTS screenshot_tags(
        image_id integer NOT NULL,
        affordance integer NOT NULL,
        tagger_id VARCHAR(16) NOT NULL,
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
        tagger_id VARCHAR(16) NOT NULL,
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
        tagger_id VARCHAR(16) NOT NULL,
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


def get_connection(dev=None):
    if dev is not None:
        url = dev
    else:
        url = current_app.config['SQLALCHEMY_DATABASE_URI']
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


@click.command("init-db")
@with_appcontext
def init_db_command():
    """Clear existing data and create new tables."""
    init_db()
    click.echo("Initialized the database.")


@click.command("destroy-db")
@with_appcontext
def destroy_db_command():
    """Clear existing tables."""
    drop_all()
    click.echo("Destroyed the database.")


@click.command("test-db")
@with_appcontext
def test_db_command():
    """test insert."""
    insert_screenshot('test_game', 256, 224)
    click.echo("Inserted the database.")


def init_app(app):
    """Register database functions with the Flask app. This is called by
    the application factory.
    """
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
    app.cli.add_command(destroy_db_command)
    app.cli.add_command(test_db_command)
