from os import environ

class Config:
    """ Set Flask configuration vars from .env file. """

    # General
    TESTING = environ["TESTING"]
    FLASK_DEBUG = environ["FLASK_DEBUG"]
    SECRET_KEY = environ.get('SECRET_KEY')

    #Database
    SQLALCHEMY_DATABASE_URI = environ.get("SQLALCHEMY_DATABASE_URI")
    SQLALCHEMY_TRACK_MODIFICATIONS = environ.get("SQLALCHEMY_TRACK_MODIFICATIONS")
