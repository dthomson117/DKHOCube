import cube
import visualise
import time
import DKHO

if __name__ == "__main__":
    start_time = time.time()
    cube = cube.Cube(3)
    print(cube.to_kociemba_string())
    print(cube.solve_kociemba())
    shuffle_moves = cube.random_moves(5)
    cube.run_moves(shuffle_moves)

    dkho = DKHO.DKHO(cube)
    solve_moves = dkho.get_hof()[0]
    cube.run_moves(solve_moves)

    print("--- %s seconds ---" % (time.time() - start_time))
    print(cube)

    visualise.show_moves(shuffle_moves=shuffle_moves, solve_moves=solve_moves)


