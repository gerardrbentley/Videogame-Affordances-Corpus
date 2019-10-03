
import os
import glob
import pickle
import random
import multiprocessing

from deap import base, creator, tools
import deap.algorithms as EA
import numpy as np

random.seed(47)


def np_help(x):
    return np.frombuffer(x, dtype=np.uint8)


PICKLE_DIR = 'saved'

print('Loading Pickles from : {}'.format(PICKLE_DIR))
tile_sets = {}
for file in glob.glob(os.path.join(PICKLE_DIR, '*.tiles')):
    curr_pickle = pickle.load(open(file, 'rb'))

    file_name = os.path.split(file)[1]
    file_name = os.path.splitext(file_name)[0]
    file_name = int(file_name)
    options = {}
    for i, (tiles, y_offset, x_offset) in enumerate(curr_pickle):
        #TODO convert tiles with np.frombuffer(__, dtype=np.uint8)
        np_tiles = list(map(np_help, tiles))
        options[i] = {'tiles': np_tiles,
                      'y_offset': y_offset, 'x_offset': x_offset}
    tile_sets[file_name] = options
print(f'Loaded {len(tile_sets)} pickled tile_set options')

#individuals have length equal to num images
IND_SIZE = len(tile_sets)


def unique_concat(previous, new):
    out = previous.copy()
    dupes = 0
    for potential in new:
        is_new = True
        for seen in previous:
            if np.array_equal(potential, seen):
                # print('ARR EQUAL')
                is_new = False
                dupes += 1
                break
        if is_new:
            # print('add new')
            # print(type(out), len(out))
            out.append(potential)
            # print(type(out), len(out))
    return out, dupes
#must return tuple for fitness value


def evalTileSets(individual):
    unique_tiles = []
    running_dupes = 0
    for i, set_idx in enumerate(individual):
        img_tiles = tile_sets[i][set_idx]['tiles']
        unique_tiles, dupes = unique_concat(unique_tiles, img_tiles)
        running_dupes += dupes
    # print(
    #     f'unique: {len(unique_tiles)}, running_dupes: {running_dupes}')
    return len(unique_tiles)


def main():
    random.seed(47)
    zero_init = np.zeros((IND_SIZE,), dtype=np.uint8)
    best = evalTileSets(zero_init)
    print('zero initialized score: ', best)
    greedy_set = []
    bincount = [0, 0, 0, 0, 0]
    for i in range(IND_SIZE):
        curr_options = tile_sets[i]
        best_next = len(greedy_set) + 100
        best_idx = -1
        updated_greedy = []
        for idx in range(5):
            tile_set = curr_options[idx]['tiles']
            uniques, dupes = unique_concat(greedy_set, tile_set)
            if len(uniques) < best_next:
                # print('better, ', len(tile_set), '..', idx)
                best_idx = idx
                updated_greedy = uniques
                best_next = len(uniques)
        print(
            f'iteration {i}, prev: {len(greedy_set)}, new greed: {len(updated_greedy)}')
        bincount[best_idx] += 1
        greedy_set = updated_greedy
    return bincount, greedy_set


if __name__ == '__main__':
    b, g = main()
    print('done')
