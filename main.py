import copy

import cube
import visualise


if __name__ == "__main__":
    cube = cube.Cube(3)
    shuffle_moves = cube.shuffle(10)
    solve_moves = copy.deepcopy(shuffle_moves)
    solve_moves.reverse()
    visualise.show_moves(shuffle_moves, solve_moves)

