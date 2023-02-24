import random


class Cube:
    """
    A class for the Rubik's Cube and all corresponding methods and attributes
    """

    # White, Green, Red, Blue, Orange, Yellow
    colours = ['w', 'g', 'r', 'b', 'o', 'y']

    move_map = {'F': ['rot_front'], 'F`': ['rot_front_acw'],
                'B': ['rot_back'], 'B`': ['rot_back_acw'],
                'L': ['rot_left'], 'L`': ['rot_left_acw'],
                'R': ['rot_right'], 'R`': ['rot_right_acw'],
                'U': ['rot_up'], 'U`': ['rot_up_acw'],
                'D': ['rot_down'], 'D`': ['rot_down_acw'],
                'F2': ['rot_front', 'rot_front'],
                'B2': ['rot_back', 'rot_back'],
                'L2': ['rot_left', 'rot_left'],
                'R2': ['rot_right', 'rot_right'],
                'U2': ['rot_up', 'rot_up'],
                'D2': ['rot_down', 'rot_down'],
                }

    def __init__(self, cube_size):
        """
        Cube builder

        :param cube_size: Size of Rubik's Cube
        """

        # Handle cubesize
        if 1 < cube_size <= 20:
            self.cube_size = cube_size
        else:
            raise ValueError("Cube cubesize cannot be smaller than 2 or larger than 20")

        # Create an empty matrix of size cubesize*cubesize*6
        self.cube = [[[None for x in range(self.cube_size)] for x in range(self.cube_size)] for x in range(6)]

        # Assign colours to each face
        for i, colour in enumerate(self.colours):
            for j in range(self.cube_size):
                for k in range(self.cube_size):
                    self.cube[i][j][k] = colour

    def __str__(self):
        printstring = ""
        for face in self.cube:
            for row in face:
                printstring += str(row) + '\n'
            printstring += '\n'
        return printstring

    def reset(self):
        """
        Reset the cube to the default state
        """
        # Assign colours to each face
        for i, colour in enumerate(self.colours):
            for j in range(self.cube_size):
                for k in range(self.cube_size):
                    self.cube[i][j][k] = colour

    def shuffle(self, shuffle_amount):
        """
        Shuffles the cube randomly shuffle_amount of moves

        :param shuffle_amount: Number of moves applied to the cube
        :returns: The list of moves applied
        """
        move_list = []

        for i in range(shuffle_amount):
            move = self.moves[random.randint(0, len(self.moves) - 1)]
            if len(move)
            move_list.append(move)
            getattr(self, self.move_map[move])

        return move_list

    def run_moves(self, moves):
        return

    def is_solved(self):
        """
        Returns a boolean stating if the cube is solved or not
        :return: True if solved, else False
        """
        for face in self.cube:
            if face.count(face[0]) != len(face):
                return False
        return True

    def rot_front(self):
        """
        Rotates the side of the cube facing the solver clockwise (the front of the cube)
        """

    def rot_front_acw(self):
        """
        Rotates the side of the cube facing the solver anti-clockwise (the front of the cube)
        :return:
        """
        return

    def rot_left(self):
        """
        Rotates the side of the cube on the left clockwise relative to the solver
        """
        return

    def rot_left_acw(self):
        """
        Rotates the side of the cube on the left anti-clockwise relative to the solver
        """
        return

    def rot_right(self):
        """
        Rotates the side of the cube on the right clockwise relative to the solver
        """
        return

    def rot_right_acw(self):
        """
        Rotates the side of the cube on the right anti-clockwise relative to the solver
        """
        return

    def rot_back(self):
        """
        Rotates the side of the cube on the back clockwise relative to the solver
        """
        return

    def rot_back_acw(self):
        """
        Rotates the side of the cube on the back anti-clockwise relative to the solver
        """
        return

    def rot_down(self):
        """
        Rotates the side of the cube on the bottom clockwise relative to the solver
        """
        return

    def rot_down_acw(self):
        """
        Rotates the side of the cube on the bottom anti-clockwise relative to the solver
        """
        return

    def rot_up(self):
        """
        Rotates the side of the cube on the top clockwise relative to the solver
        """
        return

    def rot_up_acw(self):
        """
        Rotates the side of the cube on the top anti-clockwise relative to the solver
        """
        return
