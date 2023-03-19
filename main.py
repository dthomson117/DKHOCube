import copy
import matplotlib.pyplot as plt
import cube
import visualise
import time
import DKHO

NUM_KRILL = 2000
MAX_GENERATIONS = 250
CXPB = 0.2
MUTPB = 0.7
EVAL_DEPTH = 1
MIN_MUTATE = 1
MAX_MUTATE = 5
SELECTION_SIZE = 4
PARSIMONY_SIZE = 1.2
LAMBDA = 2000

if __name__ == "__main__":

    for shuffle_amount in range(1, 20):
        start_time = time.time()
        cube = cube.Cube(3)
        shuffle_moves = cube.random_moves(7)
        cube.run_moves(shuffle_moves)

        dkho = DKHO.DKHO(cube, NUM_KRILL, MAX_GENERATIONS, CXPB, MUTPB, EVAL_DEPTH, MIN_MUTATE, MAX_MUTATE, SELECTION_SIZE, PARSIMONY_SIZE, LAMBDA)
        hof = dkho.get_hof()
        solve_moves = []

        for solution in hof:
            test_cube = copy.deepcopy(cube)
            test_cube.run_moves(solution)
            if test_cube.is_solved():
                solve_moves = solution
                break

        if not solve_moves:
            print("No solution found")
        cube.run_moves(solve_moves)

        print("--- %s seconds ---" % (time.time() - start_time))

    visualise.show_moves(shuffle_moves=shuffle_moves, solve_moves=solve_moves)


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