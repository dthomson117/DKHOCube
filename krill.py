import cube


class Krill:
    def __init__(self, shuffled_cube):
        self.cube = shuffled_cube
        self.moves = []

    def get_moves(self):
        return self.moves

    def add_move(self, move):
        self.moves.append(move)

    def get_cube_state(self):
        self.cube.run_moves(self.moves)
        return self.cube
