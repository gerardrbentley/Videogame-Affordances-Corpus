import os
import glob

import numpy as np
''' Lists all png images with file structure DIR/GAME_NAME/img/0.png'''


def get_image_files(dir=os.path.join('..', '..', 'affordances_corpus', 'games')):
    image_files = []
    for game in list_games(dir):
        per_game_files = glob.glob(os.path.join(dir, game, 'img', '*.png'))
        image_files.append(per_game_files)
    return image_files


def try_get_label(image_file=os.path.join('..', '..', 'affordances_corpus', 'games', 'loz', 'img', '0.png')):
    label_file = image_file.replace('img', 'label').replace('png', 'npy')
    if os.path.isfile(label_file):
        print('Label File Found')
        match_all_old = False
        stacked_tensor = np.load(label_file)
    else:
        print('New Label File')
        match_all_old = True
        stacked_tensor = np.full(
            [224, 256, 9], fill_value=0.5)
    return stacked_tensor


def list_games(dir=os.path.join('..', '..', 'affordances_corpus', 'games')):
    games = next(os.walk(dir))[1]
    return games
