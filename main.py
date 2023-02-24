import cube
import visualise


if __name__ == "__main__":
    cube = cube.Cube(3)
    print(cube.is_solved())
    visualise.main(['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'])

