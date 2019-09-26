from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
# Importing Tag and Base classes from database_setup.py
from database_setup import Base, Annotation, Sprite_info, Tile, Tile_info, Sprite, Image_info

engine = create_engine('sqlite:///vgac_test.db')

# Bind the engine to the metadata of the Base class so that the
# declaratives can be accessed through a DBSession instance
Base.metadata.bind = create_engine

DBSession = sessionmaker(bind=engine)

# A DBSession() instance establishes all conversations with the database
# and represents a "staging zone" for all the objects loaded into the
# database session object.
session = DBSession()
