import cube
import visualise
import time

if __name__ == "__main__":
    start_time = time.time()
    cube = cube.Cube(3)

    shuffle_moves = cube.random_moves(25)
    cube.run_moves(shuffle_moves)

    kociemba_string = cube.to_kociemba_string()

    solve_moves = cube.solve_kociemba()
    cube.run_moves(solve_moves)

    print("--- %s seconds ---" % (time.time() - start_time))
    print(cube)

    visualise.show_moves(shuffle_moves=shuffle_moves, solve_moves=solve_moves)


