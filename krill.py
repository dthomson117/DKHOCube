import cube
import copy


class Krill:
    def __init__(self, shuffled_cube):
        self.shuffled_cube = shuffled_cube
        self.krill_move_cube = copy.copy(shuffled_cube)
        self.moves = []

    def get_moves(self):
        return self.moves

    def add_move(self, move):
        self.moves.append(move)

    def get_cube_state(self):
        self.krill_move_cube = copy.copy(self.shuffled_cube)
        self.krill_move_cube.run_moves(self.moves)
        return self.krill_move_cube
