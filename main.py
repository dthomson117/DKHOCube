import copy

import cube
import visualise


if __name__ == "__main__":
    cube = cube.Cube(3)
    cube.rot_right()
    print(cube)

    moves = cube.shuffle(20)

    #visualise.show_moves(moves)

