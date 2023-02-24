import cube
import visualise


if __name__ == "__main__":
    cube = cube.Cube(3)
    print(cube)
    moves = cube.shuffle(25)
    print(moves)
    print(cube.is_solved())
    visualise.main(['R','R`'])
    print(cube.is_solved())

