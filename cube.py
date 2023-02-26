import random
import numpy as np


def rotate_matrix(matrix, cw):
    """
    Rotate a matrix

    :param matrix: Matrix to rotate
    :param cw: Rotate clockwise if True or anti-clockwise if False
    :return: A rotated matrix
    """
    if cw:
        return list(zip(*matrix[::-1]))
    else:
        return list(zip(*matrix))[::-1]


class Cube:
    """
    A class for the Rubik's Cube and all corresponding methods and attributes

    A cube is the following colours:
            |***********|
            |**W**W**W**|
            |***********|
            |**W**W**W**|
            |***********|
            |**W**W**W**|
            |***********|
            |***********|
 ***********|***********|***********|***********
 **O**O**O**|**G**G**G**|**R**R**R**|**B**B**B**
 ***********|***********|***********|***********
 **O**O**O**|**G**G**G**|**R**R**R**|**B**B**B**
 ***********|***********|***********|***********
 **O**O**O**|**G**G**G**|**R**R**R**|**B**B**B**
 ***********|***********|***********|***********
            |**Y**Y**Y**|
            |***********|
            |**Y**Y**Y**|
            |***********|
            |**Y**Y**Y**|
            |***********|
    """

    # White, Green, Red, Blue, Orange, Yellow
    colours = ['w', 'o', 'g', 'r', 'b', 'y']

    move_map = {
        # Standard moves
        'F': ['rot_front'], 'F`': ['rot_front_acw'],
        'B': ['rot_back'], 'B`': ['rot_back_acw'],
        'L': ['rot_left'], 'L`': ['rot_left_acw'],
        'R': ['rot_right'], 'R`': ['rot_right_acw'],
        'U': ['rot_up'], 'U`': ['rot_up_acw'],
        'D': ['rot_down'], 'D`': ['rot_down_acw'],

        # Double moves
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
        self.cube = np.array([[[None for x in range(self.cube_size)] for x in range(self.cube_size)] for x in range(6)])

        # Assign colours to each face
        for i, colour in enumerate(self.colours):
            for j in range(self.cube_size):
                for k in range(self.cube_size):
                    self.cube[i][j][k] = colour

    def __str__(self):
        """
        Makes the print string nicer :)

        :return: A nice print string :)
        """
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
            move = random.choice(list(self.move_map.keys()))
            move_list.append(move)
            self.run_moves(move_list)

        return move_list

    def run_moves(self, moves):
        """
        Runs the given moves

        :param moves: List of moves
        """
        for move in moves:
            if len(self.move_map[move]) > 1:
                for m in self.move_map[move]:
                    getattr(self, m)
            else:
                getattr(self, self.move_map[move][0])

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
        # Rotate the front face (G)
        self.cube[3] = list(zip(*self.cube[3][::-1]))

        # W -> R -> Y -> O -> W
        self.cube[0][2], self.cube[3][:, 0], self.cube[5][0], self.cube[1][:, 2] = \
            self.cube[3][:, 0], self.cube[5][0], self.cube[1][:, 2], self.cube[0][2]

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
