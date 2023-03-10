import kociemba
import cube
import visualise

if __name__ == "__main__":
    cube = cube.Cube(3)

    shuffle_moves = cube.random_moves(10)
    cube.run_moves(shuffle_moves)

    kociemba_string = cube.to_kociemba_string()

    solve_moves = kociemba.solve(kociemba_string).split()
    cube.run_moves(solve_moves)

    visualise.show_moves(shuffle_moves=shuffle_moves, solve_moves=solve_moves)
