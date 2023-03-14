from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import random
import multiprocessing

import cube
import krill

NUM_KRILL = 25

pool = multiprocessing.Pool()
creator.create("Fitness", base.Fitness, weights=(-1.0))
creator.create("Individual", krill.Krill, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("map", pool.map)
toolbox.register("attr_item", cube.random_moves, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_item, NUM_KRILL)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", benchmarks.kursawe)




def evaluate_krill(individual, scrambled_cube):
    return