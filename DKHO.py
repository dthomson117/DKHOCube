from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import multiprocessing

pool = multiprocessing.Pool()

toolbox = base.Toolbox()
toolbox.register("map", pool.map)
