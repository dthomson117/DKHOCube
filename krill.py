import cube

class krill:
    def __init__(self, krillcube):
        self.cube = krillcube
        self.moves = []

    def get_moves(self):
        return self.moves

    def add_move(self, move):
        self.cube.run_moves(move)
        self.moves.append(move)
