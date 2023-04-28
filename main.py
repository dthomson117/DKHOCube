import copy
import matplotlib.pyplot as plt
import cube
import visualise
import time
import DKHO

NUM_KRILL = 50
MAX_GENERATIONS = 150
CXPB = 0.3
MUTPB = 0.6
INDPB = 0.5
EVAL_DEPTH = 1
SELECTION_SIZE = 3


PARSIMONY_SIZE = 1.5
LAMBDA = 50

if __name__ == "__main__":
    average_amount = 5
    cubes = {}
    start_time = time.time()
    cube_to_solve = cube.Cube(3)
    shuffle_moves = cube_to_solve.random_moves(50)
    cube_to_solve.run_moves(shuffle_moves)

    dkho = DKHO.DKHO(cube_to_solve, NUM_KRILL, MAX_GENERATIONS, CXPB, MUTPB, INDPB, EVAL_DEPTH, SELECTION_SIZE, PARSIMONY_SIZE, LAMBDA)
    hof = dkho.get_hof()
    solve_moves = []

    for solution in hof:
        test_cube = copy.deepcopy(cube_to_solve)
        test_cube.run_moves(solution)
        if test_cube.is_solved():
            solve_moves = solution
            break

    if not solve_moves:
        print("No solution found")
    cube_to_solve.run_moves(solve_moves)

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

def test_parameters():
    return