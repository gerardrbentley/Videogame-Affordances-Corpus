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

import db
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


@bp.route("/image")
def get_image_to_tag():
    """Show an image and thumbnail to tag."""
    image_file = os.path.join(
        '..', 'affordances_corpus', 'games', 'loz', 'img', '0.png')

    image_cv, image = P.load_image(image_file)
    image_data = b64_string(image)
    logger.debug('Image encoded')

    unique_tiles = P.find_unique_tiles(image_cv, ui_height=56)
    map_dict(encode_tile_from_dict, unique_tiles)
    logger.debug('Tiles encoded')

    tags = P.load_label(image_file)
    tag_images = P.numpy_to_images(tags)
    map_dict(b64_string, tag_images)
    output = {
        'image': image_data,
        'tiles': unique_tiles,
        'tag_images': tag_images
    }
    logger.debug('base route ok')
    return jsonify({'output': output})


@bp.route("/image/<string:game>/<string:num>")
def get_specific_image(game, num):
    """Show an image and thumbnail to tag."""
    image_file = os.path.join(
        BASE_DIR, game, 'img', '{}.png'.format(num))

    image_cv, image = P.load_image(image_file)
    logger.debug('image loaded')
    image_data = b64_string(image)
    logger.debug('image encoced')
    unique_tiles = P.find_unique_tiles(image_cv, ui_height=56)
    logger.debug('tiles found')
    map_dict(encode_tile_from_dict, unique_tiles)
    logger.debug('tiles encoced')
    tags = P.load_label(image_file)
    tag_images = P.numpy_to_images(tags)
    map_dict(b64_string, tag_images)
    output = {
        'image': image_data,
        'tiles': unique_tiles,
        'tag_images': tag_images
    }
    logger.debug('specific route ok')
    return jsonify({'output': output})


@bp.route("/tile", methods=['POST'])
def save_tile_affordances():
    """Save affordances for a certain tile"""

    affordances = request.form['affordances']
    tile = request.form['tile']
    tagger = request.form['tagger_id']
    #TODO: send tile_id to client then back, not full tile

    #TODO: save tile image with affordances

    output = {
        'Success': True,
        'Response': 200,
    }

    return output


@bp.route("/affordances", methods=['POST'])
def save_blob_affordances():
    """Save a blob for one affordance for one image"""

    affordance = request.form['affordance']
    image = request.form['image_id']
    blob = request.form['affordance_blob']
    tagger = request.form['tagger_id']

    #TODO: save in database

    output = {
        'Success': True,
        'Response': 200,
    }

    return output


#
# def get_post(id, check_author=True):
#     """Get a post and its author by id.
#
#     Checks that the id exists and optionally that the current user is
#     the author.
#
#     :param id: id of post to get
#     :param check_author: require the current user to be the author
#     :return: the post with author information
#     :raise 404: if a post with the given id doesn't exist
#     :raise 403: if the current user isn't the author
#     """
#     post = (
#         get_db()
#         .execute(
#             "SELECT p.id, title, body, created, author_id, username"
#             " FROM post p JOIN user u ON p.author_id = u.id"
#             " WHERE p.id = ?",
#             (id,),
#         )
#         .fetchone()
#     )
#
#     if post is None:
#         abort(404, "Post id {0} doesn't exist.".format(id))
#
#     if check_author and post["author_id"] != g.user["id"]:
#         abort(403)
#
#     return post
#
#
# @bp.route("/create", methods=("GET", "POST"))
# @login_required
# def create():
#     """Create a new post for the current user."""
#     if request.method == "POST":
#         title = request.form["title"]
#         body = request.form["body"]
#         error = None
#
#         if not title:
#             error = "Title is required."
#
#         if error is not None:
#             flash(error)
#         else:
#             db = get_db()
#             db.execute(
#                 "INSERT INTO post (title, body, author_id) VALUES (?, ?, ?)",
#                 (title, body, g.user["id"]),
#             )
#             db.commit()
#             return redirect(url_for("blog.index"))
#
#     return render_template("blog/create.html")
#
#
# @bp.route("/<int:id>/update", methods=("GET", "POST"))
# @login_required
# def update(id):
#     """Update a post if the current user is the author."""
#     post = get_post(id)
#
#     if request.method == "POST":
#         title = request.form["title"]
#         body = request.form["body"]
#         error = None
#
#         if not title:
#             error = "Title is required."
#
#         if error is not None:
#             flash(error)
#         else:
#             db = get_db()
#             db.execute(
#                 "UPDATE post SET title = ?, body = ? WHERE id = ?", (
#                     title, body, id)
#             )
#             db.commit()
#             return redirect(url_for("blog.index"))
#
#     return render_template("blog/update.html", post=post)
#
#
# @bp.route("/<int:id>/delete", methods=("POST",))
# @login_required
# def delete(id):
#     """Delete a post.
#
#     Ensures that the post exists and that the logged in user is the
#     author of the post.
#     """
#     get_post(id)
#     db = get_db()
#     db.execute("DELETE FROM post WHERE id = ?", (id,))
#     db.commit()
#     return redirect(url_for("blog.index"))
