import os
import logging

from flask import Flask

from flask import render_template


def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        # a default secret that should be overridden by instance config
        SECRET_KEY="dev",

        # TODO: database connection / fake
        # store the database in the instance folder
        # DATABASE=os.path.join(app.instance_path, "flaskr.sqlite"),
    )

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('werkzeug').setLevel(logging.INFO)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.update(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route("/test")
    def hello():
        return render_template('base.html')

    # TODO: database connection / fake
    # register the database commands
    # from flaskr import db
    #
    # db.init_app(app)

    # apply the blueprints to the app
    from vgac_tagging import tagger

    app.register_blueprint(tagger.bp)
    app.add_url_rule("/", endpoint="index")

    return app
