import os
import logging
from dotenv import load_dotenv
from flask import Flask

from flask import render_template
from flask import send_from_directory
from flask_sqlalchemy import SQLAlchemy
import db
import tagger

load_dotenv()


def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__, instance_relative_config=True, static_url_path='')
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

    @app.route("/")
    def test_serve():
        return render_template('served.html')

    @app.route("/testjs")
    def static_js():
        return send_from_directory('templates/js/', 'testscript.js')

    #TODO: get from env variables for docker
    POSTGRES_URL = os.getenv('POSTGRES_URL')
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
    POSTGRES_DB = os.getenv('POSTGRES_DB')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT')

    DB_URL = 'postgresql+psycopg2://{}:{}@{}:{}/{}'.format(
        POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_URL, POSTGRES_PORT, POSTGRES_DB)

    # POSTGRES_URL = 'localhost'
    # POSTGRES_USER = 'gbkh2015'
    # POSTGRES_PASS = 'dev'
    # POSTGRES_DB = 'affordances_db'
    #
    # DB_URL = 'postgresql+psycopg2://{}:{}@{}/{}'.format(
    #     POSTGRES_USER, POSTGRES_PASS, POSTGRES_URL, POSTGRES_DB)

    app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
    # silence the deprecation warning
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # TODO: database connection / fake
    # register the database commands

    #db.init_app(app)

    # apply the blueprints to the app

    app.register_blueprint(tagger.bp)
    app.add_url_rule("/", endpoint="index")

    return app
