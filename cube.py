import random


class Cube:
    """
    A class for the Rubik's Cube and all corresponding methods and attributes
    """

    # White, Green, Red, Blue, Orange, Yellow
    colours = ['w', 'g', 'r', 'b', 'o', 'y']
    moves = []

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
        """
        return

    def rot_front(self, cw):
        """
        Rotates the side of the cube facing the solver (the front of the cube)

        :param cw: True if the rotation is clockwise, otherwise False
        """
        return

    def rot_left(self, cw):
        """
        Rotates the side of the cube on the left relative to the solver

        :param cw: True if the rotation is clockwise, otherwise False
        """
        return

    def rot_right(self, cw):
        """
        Rotates the side of the cube on the right relative to the solver

        :param cw: True if the rotation is clockwise, otherwise False
        """
        return

    def rot_back(self, cw):
        """
        Rotates the side of the cube on the back relative to the solver

        :param cw: True if the rotation is clockwise, otherwise False
        """
        return

    def rot_down(self, cw):
        """
        Rotates the side of the cube on the bottom relative to the solver

        :param cw: True if the rotation is clockwise, otherwise False
        """
        return

    def rot_up(self, cw):
        """
        Rotates the side of the cube on the top relative to the solver

        :param cw: True if the rotation is clockwise, otherwise False
        """
        return
