class Cube:
    """
    A class for the Rubik's Cube and all corresponding methods and attributes
    """

    # White, Green, Red, Blue, Orange, Yellow
    colours = ['w', 'g', 'r', 'b', 'o', 'y']

    def __init__(self, cubesize):
        """
        Cube builder

        :param cubesize: Size of Rubik's Cube
        """

        self.cubesize = cubesize

        # Create an empty array of size cubesize*cubesize*6
        self.cube = [[[None for x in range(self.cubesize)] for x in range(self.cubesize)] for x in range(6)]

        # Assign colours to each face
        for i, colour in enumerate(self.colours):
            for j in range(self.cubesize):
                for k in range(self.cubesize):
                    self.cube[i][j][k] = colour

    def reset(self):
        """
        Reset the cube to the default state

        :return:
        """
        # Assign colours to each face
        for i, colour in enumerate(self.colours):
            for j in range(self.cubesize):
                for k in range(self.cubesize):
                    self.cube[i][j][k] = colour

