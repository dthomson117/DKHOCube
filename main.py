import copy

import cube
import visualise


if __name__ == "__main__":
    cube = cube.Cube(3)

    moves = ['U', 'D']

    cube.run_moves(moves)

    print(cube)

    visualise.show_moves(moves)

