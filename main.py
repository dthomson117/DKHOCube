import cube
import visualise


if __name__ == "__main__":
    cube = cube.Cube(3)
    moves = cube.shuffle(25)
    print(moves)
    print(cube.is_solved())
    visualise.main(moves)
    print(cube.is_solved())

