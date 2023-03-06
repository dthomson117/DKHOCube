import copy

import cube
import visualise

if __name__ == "__main__":
    cube = cube.Cube(3)

    moves = cube.random_moves(50)
    cube.run_moves(moves)

    print(moves)
    print(cube)

    visualise.show_moves(moves)