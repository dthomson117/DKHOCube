import copy

import cube
import visualise


if __name__ == "__main__":
    cube = cube.Cube(3)
    cube.rot_front()
    print(cube)

    visualise.show_moves(['F'])

