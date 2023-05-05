import copy
import math

import matplotlib.pyplot as plt
import pandas

import cube
# import visualise
import time
import DKHO
import json
import random
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy import stats
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


def test_optimality(cubes_to_solve, NUM_KRILL=100, CXPB=0, MUTPB=0, INDPB=0,
                    SELECTION_SIZE=2, PARSIMONY_SIZE=1.1, LAMBDA=100):
    results = []
    SELECTION_SIZE = int(SELECTION_SIZE)
    NUM_KRILL = int(NUM_KRILL)
    LAMBDA = int(LAMBDA)
    for cube in cubes_to_solve:
        start_time = time.time()
        dkho = DKHO.DKHO(cube[0], NUM_KRILL, 250, CXPB, MUTPB, INDPB, 1, SELECTION_SIZE,
                         PARSIMONY_SIZE, LAMBDA)
        finish_time = time.time() - start_time
        indv = dkho.get_hof()[0]

        results.append((indv, cube, finish_time))

    loss = sum([get_optimality(ind, cube[0], time) for ind, cube, time in results])

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


def create_test_cubes(n, linear=False, fixed_shuffle=5):
    cubes = []
    for i in range(n):
        if linear:
            shuffle_amount = i + 1
        elif not linear and fixed_shuffle:
            shuffle_amount = fixed_shuffle
        else:
            shuffle_amount = random.randint(0, 26)
        shuffle_cube = cube.Cube(3, shuffle_amount=shuffle_amount)
        cubes.append((shuffle_cube, shuffle_amount))
    return cubes


def bayesian_opt():
    pbounds = {'CXPB': (0.1, 0.3), 'MUTPB': (0.1, 0.7), 'INDPB': (0.1, 1), 'PARSIMONY_SIZE': (1.1, 1.9),
               'SELECTION_SIZE': (2, 10), 'NUM_KRILL': (5, 250), 'LAMBDA': (5, 250)}
    logger = JSONLogger(path="./logs-run-2.json")
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


def test_time(best, cubes_to_solve):
    results = []

    for shuffled_cube in cubes_to_solve:
        dkho = DKHO.DKHO(shuffled_cube[0], int(best['NUM_KRILL']), 250, best['CXPB'], best['MUTPB'], best['INDPB'], 1,
                         int(best['SELECTION_SIZE']), best['PARSIMONY_SIZE'], int(best['LAMBDA']))

        results.append((dkho.get_hof(), dkho.get_logbook(), dkho.get_time(), shuffled_cube))

    with open('time_opt_results.pkl', 'wb') as file:
        pickle.dump(results, file)


def get_best_params():
    iterations = read_json_bo()
    highest_target = 0
    for line in iterations:
        if line['target'] > highest_target:
            highest_target = line['target']
            best = line

    return best['params']


def read_json_bo():
    iterations = []

    with open("logs.json") as f:
        for obj in f:
            optDict = json.loads(obj)
            iterations.append(optDict)

    return iterations


def read_pickle(filename):
    with open(filename, 'rb') as file:
        results = pickle.load(file)

    return results


def extract_pickle_data(data):
    return


def check_failrate():
    data = read_pickle('comp_results.pkl')
    fitness_min = []
    for hof, log, tt in data[0]:
        hof = hof
        log = log
        tt = tt

        fitness_min.append(log.chapters['fitness'].select('min'))

    count_5, count_10, count_15, count_20, count_25 = 0, 0, 0, 0, 0
    sum_5, sum_10, sum_15, sum_20, sum_25 = 0, 0, 0, 0, 0
    for i, run in enumerate(fitness_min):
        if i < 50:
            if min(run) > 0:
                count_5 += 1
            sum_5 += 1
        elif i < 100:
            if min(run) > 0:
                count_10 += 1
            sum_10 += 1
        elif i < 150:
            if min(run) > 0:
                count_15 += 1
            sum_15 += 1
        elif i < 200:
            if min(run) > 0:
                count_20 += 1
            sum_20 += 1
        else:
            if min(run) > 0:
                count_25 += 1
            sum_25 += 1

    print("5: " + str(count_5) + '/ 50')
    print("10: " + str(count_10) + '/ 50')
    print("15: " + str(count_15) + '/ 50')
    print("20: " + str(count_20) + '/ 50')
    print("25: " + str(count_25) + '/ 50')


def plot_dkho_vs_koc_2():
    data = read_pickle('time_opt_results.pkl')
    info = []

    for hof, log, tt, cube_data in data:
        hof = hof
        log = log
        tt = tt

        run_indv = hof[0]
        shuffled_cube = cube_data

        info.append((run_indv, shuffled_cube))

    kociemba_lengths = []
    dkho_lengths = []
    scramble_lengths = []
    for indv, cubes in info:
        if indv.fitness.values[0] != 0:
            continue
        copy_cube = copy.deepcopy(cubes[0])
        copy_cube.run_moves(indv)
        if not copy_cube.is_solved():
            continue

        best = 100
        for i in range(5):
            solution = cubes[0].solve_kociemba()
            if solution == ['']:
                best = 0
                break
            if len(solution) < best:
                best = len(solution)
        kociemba_lengths.append(best)
        dkho_lengths.append(len(indv))
        scramble_lengths.append(cubes[1])

    second_plot_data = pandas.DataFrame([], columns=['Kociemba Length', 'DKHO Length'])
    second_plot_data['Kociemba Length'] = kociemba_lengths
    second_plot_data['DKHO Length'] = dkho_lengths
    second_plot_data['Scramble Length'] = scramble_lengths
    second_plot_data['Difference'] = second_plot_data['Kociemba Length'] - second_plot_data['DKHO Length']
    colours = [cubes[1] for indv, cubes in info if indv.fitness.values[0] == 0]
    straight_line = pandas.DataFrame(np.arange(0, 25, 1))

    second_plot_data[['Kociemba Length', 'DKHO Length']].to_csv(path_or_buf='scramble-length.csv', index=False)

    # sns.scatterplot(second_plot_data, x="Scramble Length", y="Difference",
    # legend='brief').set(title="DKHO vs Kociemba Solution Lengths (w/ outliers)")

    # plt.legend(labels=['Solution Length'])
    # sns.lineplot(data=straight_line, legend=False)
    # plt.show()

    sns.set_theme(style="whitegrid")
    sns.set_palette(sns.color_palette("flare", 27))
    print("w/: " + str(second_plot_data['Difference'].mean()))
    histogram_data = second_plot_data.groupby('Scramble Length')['Difference'].sum().reset_index()
    histogram_data.columns = ['Scramble Length', 'Difference']

    sns.catplot(histogram_data, x="Scramble Length", y='Difference',
                legend=False, errorbar='sd', kind='bar', height=8.27, aspect=11.7 / 8.27).set(
        title="DKHO vs Kociemba Solution Lengths (w/ outliers)")

    # sns.lineplot(data=straight_line, legend=False)

    plt.savefig('./images/DKHO vs Kociemba Solution Lengths (w outliers).png', dpi=300)
    plt.subplots_adjust(top=0.9)
    plt.show()

    second_plot_data = second_plot_data[
        (np.abs(stats.zscore(second_plot_data.select_dtypes(exclude='object'))) < 2).all(axis=1)]

    second_plot_data[['Kociemba Length', 'DKHO Length']].to_csv(path_or_buf='scramble-length-wo-outliers.csv',
                                                                index=False)

    print("w/o: " + str(second_plot_data['Difference'].mean()))
    histogram_data = second_plot_data.groupby('Scramble Length')['Difference'].sum().reset_index()
    histogram_data.columns = ['Scramble Length', 'Difference']

    sns.catplot(histogram_data, x="Scramble Length", y='Difference',
                legend=False, errorbar='sd', kind='bar', height=8.27, aspect=11.7 / 8.27).set(
        title="DKHO vs Kociemba Solution Lengths (w/o outliers)")

    # sns.lineplot(data=straight_line, legend=False)
    plt.subplots_adjust(top=0.9)
    plt.savefig('./images/DKHO vs Kociemba Solution Lengths (wo outliers).png', dpi=300)

    plt.show()


def plot_solve_lengths_2():
    data = read_pickle('time_opt_results.pkl')
    info = []

    for hof, log, tt, cube_data in data:
        hof = hof
        log = log
        tt = tt

        run_indv = hof[0]
        shuffled_cube = cube_data

        info.append((run_indv, shuffled_cube))

    plot_data = pandas.DataFrame([(y[1], len(x)) for x, y in info], columns=['Shuffle Length', 'Solution Length'])
    plot_data['Outlier'] = np.where((np.abs(stats.zscore(plot_data['Solution Length'])) < 2), False, True)
    print(len(plot_data))
    sns.set_theme(style="whitegrid")
    mean_data = plot_data.groupby('Shuffle Length', as_index=False)['Solution Length'].mean()
    sns.scatterplot(plot_data, x="Shuffle Length", y="Solution Length", hue='Outlier', style='Outlier', legend='auto').set(
        title="Solution Length with Average (w/ Outliers)")
    sns.lineplot(mean_data['Solution Length'], color='red', legend='full')
    plt.legend(title="Outliers")
    plt.savefig('./images/Solution Length with Average (w Outliers).png', dpi=300)
    plt.show()

    plot_data = pandas.DataFrame([(y[1], len(x)) for x, y in info], columns=['Shuffle Length', 'Solution Length'])
    plot_data = plot_data[(np.abs(stats.zscore(plot_data['Solution Length'])) < 2)]
    print(len(plot_data))

    mean_data = plot_data.groupby('Shuffle Length', as_index=False)['Solution Length'].mean()
    sns.scatterplot(plot_data, x="Shuffle Length", y="Solution Length").set(
        title="Solution Length with Average (w/o Outliers)")
    sns.lineplot(mean_data['Solution Length'], color='red', legend='auto')
    plt.legend(labels=['Solution Length', 'Average'])
    plt.savefig('./images/Solution Length with Average (wo Outliers).png', dpi=300)
    plt.show()


def plot_time():
    data = read_pickle('time_opt_results.pkl')
    info = []

    for hof, log, tt, cube_data in data:
        hof = hof
        log = log
        dkho_tt = tt
        copy_cube = copy.deepcopy(cube_data[0])
        start_time = time.time()
        copy_cube.solve_kociemba()
        kociemba_time = time.time() - start_time

        if hof[0].fitness.values[0] == 0:
            info.append((dkho_tt, kociemba_time, cube_data[1]))

    plot_data = pandas.DataFrame(info, columns=['DKHO', 'Kociemba', 'Shuffle Length'])
    df = plot_data.melt(id_vars=['Shuffle Length'], value_name='Time Taken', var_name='Algorithm')
    df = df[df['Algorithm'] != 'Kociemba']
    avg = df['Time Taken'].mean()
    print(avg)
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    sns.set_theme(style="whitegrid")

    sns.pointplot(data=df, x='Shuffle Length', y='Time Taken', errorbar='ci', capsize=.5, color='b', markers='o', errwidth=1.5, linestyles=['--'], scale=.6, label='DKHO Average')

    plt.subplots_adjust(left=0.083, right=0.96, top=0.912, bottom=0.098)
    plt.legend()
    plt.ylabel("Time Taken (s)")
    plt.title("Solution Time w/ Outliers")
    plt.savefig("./images/Solution Time w Outliers.png", dpi=300)
    plt.show()

    clean_plot_data = pandas.DataFrame(info, columns=['DKHO', 'Kociemba', 'Shuffle Length'])
    clean_plot_data = clean_plot_data[(np.abs(stats.zscore(clean_plot_data['DKHO'])) < 2)]
    df = clean_plot_data.melt(id_vars=['Shuffle Length'], value_name='Time Taken', var_name='Algorithm')
    df = df[df['Algorithm'] != 'Kociemba']
    avg = df['Time Taken'].mean()
    print(avg)

    sns.pointplot(data=df, x='Shuffle Length', y='Time Taken', errorbar='ci', capsize=.3, color='b', markers='o', errwidth=1.5, linestyles=['--'], scale=.6, label='DKHO Average')

    plt.subplots_adjust(left=0.083, right=0.96, top=0.912, bottom=0.098)
    plt.legend()
    plt.title("Solution Time w/o Outliers")
    plt.ylabel("Time Taken (s)")
    plt.savefig("./images/Solution Time wo Outliers.png", dpi=300)
    plt.show()

def plot_time_points():
    data = read_pickle('time_opt_results.pkl')
    info = []

    for hof, log, tt, cube_data in data:
        hof = hof
        log = log
        dkho_tt = tt
        copy_cube = copy.deepcopy(cube_data[0])
        start_time = time.time()
        copy_cube.solve_kociemba()
        kociemba_time = time.time() - start_time

        if hof[0].fitness.values[0] == 0:
            info.append((dkho_tt, kociemba_time, cube_data[1]))

    plot_data = pandas.DataFrame(info, columns=['DKHO', 'Kociemba', 'Shuffle Length'])
    df = plot_data.melt(id_vars=['Shuffle Length'], value_name='Time Taken', var_name='Algorithm')
    df = df[df['Algorithm'] != 'Kociemba']
    avg = df['Time Taken'].mean()
    df['Outlier'] = np.where((np.abs(stats.zscore(df['Time Taken'])) < 2), False, True)
    print(avg)
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    sns.set_theme(style="whitegrid")

    sns.scatterplot(data=df, x='Shuffle Length', y='Time Taken', hue='Outlier', style='Outlier', legend='brief')

    plt.subplots_adjust(left=0.083, right=0.96, top=0.912, bottom=0.098)
    plt.legend(title="Outlier")
    plt.ylabel("Time Taken (s)")
    plt.title("Solution Time Points w/ Outliers")
    plt.savefig("./images/Solution Time Points w Outliers.png", dpi=300)
    plt.show()

    clean_plot_data = pandas.DataFrame(info, columns=['DKHO', 'Kociemba', 'Shuffle Length'])
    clean_plot_data = clean_plot_data[(np.abs(stats.zscore(clean_plot_data['DKHO'])) < 2)]
    df = clean_plot_data.melt(id_vars=['Shuffle Length'], value_name='Time Taken', var_name='Algorithm')
    df = df[df['Algorithm'] != 'Kociemba']
    avg = df['Time Taken'].mean()
    print(avg)

    sns.scatterplot(data=df, x='Shuffle Length', y='Time Taken', hue='Algorithm', style='Algorithm', legend='brief')

    plt.subplots_adjust(left=0.083, right=0.96, top=0.912, bottom=0.098)
    plt.legend()
    plt.title("Solution Time Points w/o Outliers")
    plt.ylabel("Time Taken (s)")
    plt.savefig("./images/Solution Time Points wo Outliers.png", dpi=300)
    plt.show()


def implementation_comparison(best):
    scrambles = []
    scrambles.append("F2 L D2 B2 F".split(" "))
    scrambles.append("B U' R2 B F'".split(" "))
    scrambles.append("R2 B2 D F R2".split(" "))
    scrambles.append("U D2 F' U2 R' B2 L D' F2 U2 B2 R' F2 U L'".split(" "))
    scrambles.append("D B L U B L2 U2 F' R' L' D2 R2 F U2 L'".split(" "))
    scrambles.append("B2 L2 U B F2 D' U' R2 D B' U F' R' B' L".split(" "))
    scrambles.append("L2 D2 U' L2 F' R' U' F' R2 L U2 R2 F' U' R2 F2 D B U' D2 B' L F U' R'".split(" "))
    scrambles.append("F R L D B' F' D L R D U2 L D' R F U' D' L F' D U R D L' F2".split(" "))
    scrambles.append("R U2 L' D B' R F L' B2 D2 F' L2 B' U L2 B R D' U' R2 B2 D L2 F U".split(" "))

    for scramble in scrambles:
        shuffle_cube = cube.Cube(3)
        shuffle_cube.run_moves(scramble)

        # dkho = DKHO.DKHO(shuffle_cube, int(best['NUM_KRILL']), 250, best['CXPB'], best['MUTPB'], best['INDPB'], 1,
        # int(best['SELECTION_SIZE']), best['PARSIMONY_SIZE'], int(best['LAMBDA']))

        # print(dkho.get_time())

        start = time.time()
        shuffle_cube.solve_kociemba()
        end = time.time() - start

        print(end)


if __name__ == "__main__":
    # cubes_to_solve = create_test_cubes(25, linear=True)
    # best_args = bayesian_opt(cubes_to_solve)
    # print(best_args)

    # plot_optimizer()

    # plot_search_space_bo()

    # plot_kde()

    # cubes_to_solve = create_test_cubes(1000, linear=False, fixed_shuffle=False)
    # test_time(get_best_params(), cubes_to_solve=cubes_to_solve)

    # plot_completeness()

    # implementation_comparison(get_best_params())

    plot_solve_lengths_2()
    #plot_time()
    #plot_dkho_vs_koc_2()
    #plot_time_points()
