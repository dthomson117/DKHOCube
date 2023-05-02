import copy
import math

import matplotlib.pyplot as plt
import cube
# import visualise
import time
import DKHO
import json
import random
import numpy as np
import pandas as pd
import seaborn as sns
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import SequentialDomainReductionTransformer


def plot_logbook(logbook):
    gen = logbook.select("gen")
    avgs = logbook.select("avg")
    stds = logbook.select("std")
    avgs_value = [item[0] for item in avgs]
    fig, ax = plt.subplots()
    line = ax.plot(gen, avgs_value)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (value)")
    plt.show()


"""
def test_parameters():
    combinations = [(round(x / 10, 2), round(1 - x / 10, 2)) for x in range(0, 11)]
    results = {}

    for mutpb, crosspb in combinations:
        dkho = DKHO.DKHO(cube_to_solve, NUM_KRILL, MAX_GENERATIONS, crosspb, mutpb, INDPB, EVAL_DEPTH, SELECTION_SIZE,
                         PARSIMONY_SIZE, LAMBDA)
        results[(mutpb, crosspb)] = {dkho.get_logbook()}

    print(results)
    return results

"""


def test_optimality(NUM_KRILL=100, CXPB=0, MUTPB=0, INDPB=0,
                    SELECTION_SIZE=2, PARSIMONY_SIZE=1.1, LAMBDA=100):
    results = []
    SELECTION_SIZE = int(SELECTION_SIZE)
    NUM_KRILL = int(NUM_KRILL)
    LAMBDA = int(LAMBDA)
    for cube in cubes_to_solve:
        start_time = time.time()
        dkho = DKHO.DKHO(cube, NUM_KRILL, 250, CXPB, MUTPB, INDPB, 1, SELECTION_SIZE,
                         PARSIMONY_SIZE, LAMBDA)
        finish_time = time.time() - start_time
        indv = dkho.get_hof()[0]

        results.append((indv, cube, finish_time))

    loss = sum([get_optimality(ind, cube, time) for ind, cube, time in results])

    return loss


def get_optimality(indv, shuffled_cube, time_taken):
    optimal = 9999
    for i in range(5):
        optimal_solve = shuffled_cube.solve_kociemba()
        if '' in optimal_solve:
            optimal_solve.remove('')
        if len(optimal_solve) < optimal:
            optimal = len(optimal_solve)
    score = optimal - len(indv)
    return 1 / (score + (time_taken / 100))


def create_test_cubes(n):
    cubes = []
    for i in range(n):
        shuffle_cube = cube.Cube(3)
        shuffle_amount = random.randint(0, 50)
        shuffle_cube.run_moves(shuffle_cube.random_moves(shuffle_amount))
        cubes.append((shuffle_cube, shuffle_amount))
    return cubes


def bayesian_opt():
    pbounds = {'CXPB': (0.1, 0.3), 'MUTPB': (0.1, 0.7), 'INDPB': (0.1, 1), 'PARSIMONY_SIZE': (1.1, 1.9),
               'SELECTION_SIZE': (2, 10), 'NUM_KRILL': (5, 250), 'LAMBDA': (5, 250)}
    logger = JSONLogger(path="./logs.json")
    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.1)
    optimizer = BayesianOptimization(f=test_optimality, pbounds=pbounds, random_state=1, verbose=2,
                                     bounds_transformer=bounds_transformer)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.set_gp_params(alpha=1e-3, n_restarts_optimizer=5)
    optimizer.maximize(
        init_points=5,
        n_iter=50,
    )
    plt.plot(optimizer.space.target, label="Optimizer")
    plt.legend()
    plt.show()
    return optimizer.max


def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = max(y)
    text = "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=-30")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)


def plot_optimizer():
    iterations = read_json_bo()

    targets = []
    for line in iterations:
        targets.append(line['target'])

    plt.plot(targets, label="Optimizer")
    plt.legend(loc='upper center', bbox_to_anchor=(0.9, -0.05),
               fancybox=True, shadow=True, ncol=1)
    plt.xlabel("Iterations")
    plt.ylabel("Target")
    plt.title("Bayesian Optimisation")
    plt.ylim(0, max(targets) + 10)
    plt.xlim(0, len(targets))
    annot_max(list(range(1, len(targets))), targets)
    plt.show()


def plot_search_space_bo():
    iterations = read_json_bo()
    values = []
    for i in iterations:
        values.append(list(i['params'].items()))

    x_y_values = [item for sublist in values for item in sublist]

    mutpb = [value for x, value in x_y_values if x == "MUTPB"]
    cxpb = [value for x, value in x_y_values if x == "CXPB"]
    indpb = [value for x, value in x_y_values if x == "INDPB"]

    num_krill = [value for x, value in x_y_values if x == "NUM_KRILL"]
    lambda_ = [value for x, value in x_y_values if x == "LAMBDA"]

    sel_size = [value for x, value in x_y_values if x == "SELECTION_SIZE"]
    pars_size = [value for x, value in x_y_values if x == "PARSIMONY_SIZE"]

    graph1 = {"MUTPB": mutpb, "CXPB": cxpb, "INDPB": indpb}
    graph2 = {"NUM_KRILL": num_krill, "LAMBDA": lambda_}
    graph3 = {"SELECTION_SIZE": sel_size}
    graph4 = {"PARSIMONY_SIZE": pars_size}

    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0, 0].boxplot(graph1.values(), whis=1000)
    ax[0, 0].set_xticklabels(graph1.keys())

    ax[0, 1].boxplot(graph2.values(), whis=1000)
    ax[0, 1].set_xticklabels(graph2.keys())

    ax[1, 0].boxplot(graph3.values(), whis=1000)
    ax[1, 0].set_xticklabels(graph3.keys())

    ax[1, 1].boxplot(graph4.values(), whis=1000)
    ax[1, 1].set_xticklabels(graph4.keys())

    fig.suptitle("Hyper-parameter Samples")
    fig.text(0.06, 0.5, 'Values', ha='center', va='center', rotation='vertical')
    plt.show()


def plot_kde():
    iterations = read_json_bo()
    values = []
    for i in iterations:
        values.append(list(i['params'].items()))

    x_y_values = [item for sublist in values for item in sublist]

    mutpb = [value for x, value in x_y_values if x == "MUTPB"]
    cxpb = [value for x, value in x_y_values if x == "CXPB"]
    indpb = [value for x, value in x_y_values if x == "INDPB"]

    num_krill = [value for x, value in x_y_values if x == "NUM_KRILL"]
    lambda_ = [value for x, value in x_y_values if x == "LAMBDA"]

    sel_size = [value for x, value in x_y_values if x == "SELECTION_SIZE"]
    pars_size = [value for x, value in x_y_values if x == "PARSIMONY_SIZE"]

    graph1 = {"MUTPB": mutpb, "CXPB": cxpb, "INDPB": indpb}
    graph2 = {"NUM_KRILL": num_krill, "LAMBDA": lambda_}
    graph3 = {"SELECTION_SIZE": sel_size}
    graph4 = {"PARSIMONY_SIZE": pars_size}

    df1 = pd.DataFrame(data=graph1)
    df2 = pd.DataFrame(data=graph2)
    df3 = pd.DataFrame(data=graph3)
    df4 = pd.DataFrame(data=graph4)

    sns.stripplot(df1, alpha=.5)
    sns.boxplot(df1, whis=1000, width=0.5, saturation=.7)
    plt.show()

def test_time():



def read_json_bo():
    iterations = []

    with open("logs.json") as f:
        for obj in f:
            optDict = json.loads(obj)
            iterations.append(optDict)

    return iterations


if __name__ == "__main__":
    # cubes_to_solve = create_test_cubes(25)
    # best_args = bayesian_opt()
    # print(best_args)

    # plot_optimizer()

    # plot_search_space_bo()

    #plot_kde()


