from . import db

class Tag(db.Model):
    """ Model for tables. """
    __tablename__ = 'image_info'
    img_id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.String, nullable=False)
    w = db.Column(db.Integer, nullable=False)
    h = db.Column(db.Integer, nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)

    __tablename__ = 'sprite_info'
    sprite_id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.String, nullable=False)
    w = db.Column(db.Integer, nullable=False)
    h = db.Column(db.Integer, nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)

    __tablename__ = 'tile_info'
    tile_id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.String, nullable=False)
    w = db.Column(db.Integer, nullable=False)
    h = db.Column(db.Integer, nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)


    __tablename__ = 'annotations'
    id = db.Column(db.Integer, primary_key=True)
    img_id = db.Column(db.String(64),
                        unique=False,
                        nullable=False)
    tagger_id = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    affordance = db.Column(db.Integer, nullable=False)
    tags = db.Column(db.LargeBinary, nullable=False)

    __tablename__ = 'tiles'
    id = db.Column(db.Integer, primary_key=True)
    tile_id = db.Column(db.Integer, unique=False, nullable=False)
    tagger_id = db.Column(db.String, unique=False, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    affordance1 = db.Column(db.Boolean, nullable=True)
    affordance2 = db.Column(db.Boolean, nullable=True)
    affordance3 = db.Column(db.Boolean, nullable=True)
    affordance4 = db.Column(db.Boolean, nullable=True)
    affordance5 = db.Column(db.Boolean, nullable=True)
    affordance6 = db.Column(db.Boolean, nullable=True)
    affordance7 = db.Column(db.Boolean, nullable=True)
    affordance8 = db.Column(db.Boolean, nullable=True)
    affordance9 = db.Column(db.Boolean, nullable=True)

    __tablename__ = 'sprites'
    id = db.Column(db.Integer, primary_key=True)
    sprite_id = db.Column(db.Integer, unique=False, nullable=False)
    tagger_id = db.Column(db.String, unique=False, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    affordance1 = db.Column(db.Boolean, nullable=True)
    affordance2 = db.Column(db.Boolean, nullable=True)
    affordance3 = db.Column(db.Boolean, nullable=True)
    affordance4 = db.Column(db.Boolean, nullable=True)
    affordance5 = db.Column(db.Boolean, nullable=True)
    affordance6 = db.Column(db.Boolean, nullable=True)
    affordance7 = db.Column(db.Boolean, nullable=True)
    affordance8 = db.Column(db.Boolean, nullable=True)
    affordance9 = db.Column(db.Boolean, nullable=True)
