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

import vgac_tagging.db as db_ops

import cv2
import numpy as np
# from flaskr.auth import login_required
# from flaskr.db import get_db

logger = logging.getLogger(__name__)
bp = Blueprint("tagger", __name__)

IMAGE_BASE = "data:image/png;base64,{}"


@bp.route("/")
def get_image_to_tag():
    """Show an image and thumbnail to tag."""
    # db = get_db()
    # posts = db.execute(
    #     "SELECT p.id, title, body, created, author_id, username"
    #     " FROM post p JOIN user u ON p.author_id = u.id"
    #     " ORDER BY created DESC"
    # ).fetchall()
    image = load_image()
    image_data = (base64.b64encode(image))
    logger.debug('image encoced: {}'.format(image_data))
    image_data = IMAGE_BASE.format(image_data)
    unique_tiles = find_unique_tiles(image)

    output = {
        'image': image_data,
        'tiles': unique_tiles
    }
    logger.debug('base route ok')
    return jsonify(output)


@bp.route("/apply", methods=['POST'])
def apply_affordances():
    """Apply affordances to a certain area of image"""

    affordances = request.form['affordances']
    image = request.form['image']
    tile = request.form['tile']

    #save tile image with affordances
    #mark affordances on image's label

    output = {
        'Success': True,
        'Response': 200,
        'Image': image
    }

    return output


def load_image(image_file=os.path.join('..', 'affordances_corpus', 'games', 'loz', 'img', '0.png')):
    logger.debug(image_file)
    orig_cv = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    if orig_cv.shape[2] == 4:
        orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_BGRA2BGR)
    orig_cv = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2RGB)
    return orig_cv


def gen_grid(width, height, grid_size, ui_height=0, ui_position='top', grid_offset_x=0, grid_offset_y=0):
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


GRID_SIZE = 16


def find_unique_tiles(image):
    print('Finding unique tiles in img')
    img_tiles = {}
    visited_locations = []
    tile_ctr = 0
    skip_ctr = 0
    rows, cols = gen_grid(256, 224, 16, 56)
    for r in np.unique(rows):
        for c in np.unique(cols):
            if((r, c) not in visited_locations):
                template_np = image[r:r+GRID_SIZE if r+GRID_SIZE <= 224 else 224,
                                    c:c+GRID_SIZE if c+GRID_SIZE <= 256 else 256].copy()
                template_data = base64.b64encode(template_np)
                template_data = IMAGE_BASE.format(template_data)
                res = cv2.matchTemplate(
                    template_np, image, cv2.TM_SQDIFF_NORMED)
                loc = np.where(res <= 5e-6)
                matches = list(zip(*loc[::1]))
                matches = [(y, x) for (y, x) in matches if point_on_grid(
                    x, y, cols, rows)]
                matches_int = [(int(y), int(x)) for (y, x) in matches if point_on_grid(
                    x, y, cols, rows)]
                if len(matches) != 0:
                    for match_loc in matches:
                        visited_locations.append(match_loc)
                else:
                    print(
                        'ERROR MATCHING TILE WITHIN IMAGE: (r,c) ({},{})'.format(r, c))

                img_tiles[tile_ctr] = ((template_data, matches_int))
                tile_ctr += 1
            else:
                skip_ctr += 1

    print('VISITED {} tiles, sum of unique({}) + skip({}) = {}'.format(
        len(visited_locations), len(img_tiles), skip_ctr, (len(img_tiles)+skip_ctr)))
    return img_tiles

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
