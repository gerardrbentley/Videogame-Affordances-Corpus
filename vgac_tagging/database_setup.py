import sys

#creating the mapper code
from sqlalchemy import Column, ForeignKey, Integer, String, LargeBinary, Boolean, DateTime

#for configuration and class code
from sqlalchemy.ext.declarative import declarative_base

#for creating foreign key relationship between the tables
from sqlalchemy.orm import relationship

#for configuration
from sqlalchemy import create_engine

#create declarative_base instance
Base = declarative_base()

#classes go here
class Image_info(Base):
    """ Model for tables. """
    __tablename__ = 'image_info'
    img_id = Column(Integer, primary_key=True)
    game_id = Column(String, nullable=False)
    w = Column(Integer, nullable=False)
    h = Column(Integer, nullable=False)
    data = Column(LargeBinary, nullable=False)

class Sprite_info(Base):
    __tablename__ = 'sprite_info'
    sprite_id = Column(Integer, primary_key=True)
    game_id = Column(String, nullable=False)
    w = Column(Integer, nullable=False)
    h = Column(Integer, nullable=False)
    data = Column(LargeBinary, nullable=False)

class Tile_info(Base):
    __tablename__ = 'tile_info'
    tile_id = Column(Integer, primary_key=True)
    game_id = Column(String, nullable=False)
    w = Column(Integer, nullable=False)
    h = Column(Integer, nullable=False)
    data = Column(LargeBinary, nullable=False)


class Annotation(Base):
    __tablename__ = 'annotations'
    id = Column(Integer, primary_key=True)
    img_id = Column(String(64),
                        unique=False,
                        nullable=False)
    tagger_id = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    affordance = Column(Integer, nullable=False)
    tags = Column(LargeBinary, nullable=False)

class Tile(Base):
    __tablename__ = 'tiles'
    id = Column(Integer, primary_key=True)
    tile_id = Column(Integer, unique=False, nullable=False)
    tagger_id = Column(String, unique=False, nullable=False)
    created_at = Column(DateTime, nullable=False)
    affordance1 = Column(Boolean, nullable=True)
    affordance2 = Column(Boolean, nullable=True)
    affordance3 = Column(Boolean, nullable=True)
    affordance4 = Column(Boolean, nullable=True)
    affordance5 = Column(Boolean, nullable=True)
    affordance6 = Column(Boolean, nullable=True)
    affordance7 = Column(Boolean, nullable=True)
    affordance8 = Column(Boolean, nullable=True)
    affordance9 = Column(Boolean, nullable=True)

class Sprite(Base):
    __tablename__ = 'sprites'
    id = Column(Integer, primary_key=True)
    sprite_id = Column(Integer, unique=False, nullable=False)
    tagger_id = Column(String, unique=False, nullable=False)
    created_at = Column(DateTime, nullable=False)
    affordance1 = Column(Boolean, nullable=True)
    affordance2 = Column(Boolean, nullable=True)
    affordance3 = Column(Boolean, nullable=True)
    affordance4 = Column(Boolean, nullable=True)
    affordance5 = Column(Boolean, nullable=True)
    affordance6 = Column(Boolean, nullable=True)
    affordance7 = Column(Boolean, nullable=True)
    affordance8 = Column(Boolean, nullable=True)
    affordance9 = Column(Boolean, nullable=True)

#creates a create_engine instance at bottom of the file
engine = create_engine('sqlite:///vgac-test.db')
Base.metadata.create_all(engine)
