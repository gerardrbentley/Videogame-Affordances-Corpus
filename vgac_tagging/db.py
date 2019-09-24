import os
import glob

''' Lists all png images with file structure DIR/GAME_NAME/img/0.png'''


def get_image_files(dir=os.path.join('..', 'affordances_corpus', 'games')):
    image_files = []
    for game in list_games(dir):
        per_game_files = glob.glob(os.path.join(dir, game, 'img', '*.png'))
        image_files.append(per_game_files)
    return image_files


def list_games(dir=os.path.join('..', 'affordances_corpus', 'games')):
    games = next(os.walk(dir))[1]
    return games
