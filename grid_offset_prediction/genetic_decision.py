
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
#minimization problem with single objective
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

#individuals have length of num_images, represent index selection of each's best offsets
creator.create("Individual", list, fitness=creator.FitnessMin)

#individuals have length equal to num images
IND_SIZE = len(tile_sets)
toolbox = base.Toolbox()
toolbox.register("set_idx", np.random.choice,
                 list(range(5)), p=[0.6, 0.1, 0.1, 0.1, 0.1])
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.set_idx, n=IND_SIZE)
toolbox.register("population", tools.initRepeat,
                 list, toolbox.individual)


def unique_concat(previous, new):
    out = previous
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
    return len(unique_tiles),


# OPERATORS
toolbox.register("evaluate", evalTileSets)
toolbox.register("mate", tools.cxTwoPoint)
#mut uniform int is inclusive both low and high
toolbox.register("mutate", tools.mutUniformInt, low=0, up=4, indpb=0.05)

toolbox.register("select", tools.selTournament, tournsize=3)


#MULTIPROCESSING
pool = multiprocessing.Pool()
toolbox.register("map", pool.map)


def pop_min_tiles(population):
    min = 10000
    for individual in population:
        if individual.fitness.values[0] < min:
            min = individual.fitness.values[0]
    return min


def pop_max_tiles(population):
    max = 1
    for individual in population:
        if individual.fitness.values[0] > max:
            max = individual.fitness.values[0]
    return max


def pop_avg_tiles(population):
    tot = 0.0
    for individual in population:
        tot += individual.fitness.values[0]
    return tot / len(population)


def main():
    random.seed(47)
    pop = toolbox.population(n=20)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics()
    stats.register("avg", pop_avg_tiles)
    stats.register("min", pop_min_tiles)
    stats.register("max", pop_max_tiles)
    # pop, log = EA.eaMuPlusLambda(pop, toolbox, mu=30, lambda_=70, cxpb=0.5, mutpb=0.2,
    #                              ngen=40, stats=stats, halloffame=hof, verbose=True)
    pop, log = EA.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                           ngen=40, stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof


if __name__ == '__main__':
    pop, log, hof = main()
    print(hof[0])
