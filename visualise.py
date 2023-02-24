"""This is taken from the brilliant stack overflow answer here:
https://stackoverflow.com/questions/50303616/how-to-rotate-slices-of-a-rubiks-cube-in-python-pyopengl

Modified to take a string of inputs to rotate cube, as well as initialise cube from shuffled state.

Only works for 3x3 cube
"""

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

vertices = (
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
)
edges = ((0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 7), (6, 3), (6, 4), (6, 7), (5, 1), (5, 4), (5, 7))
surfaces = ((0, 1, 2, 3), (3, 2, 7, 6), (6, 7, 5, 4), (4, 5, 1, 0), (1, 5, 7, 2), (4, 0, 3, 6))
colors = ((0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 0.5, 0), (1, 1, 0), (1, 1, 1))


class Cubie:
    def __init__(self, id, N, scale):
        self.N = N
        self.scale = scale
        self.init_i = [*id]
        self.current_i = [*id]
        self.rot = [[1 if i == j else 0 for i in range(3)] for j in range(3)]

    def isAffected(self, axis, slice, dir):
        return self.current_i[axis] == slice

    def update(self, axis, slice, dir):

        if not self.isAffected(axis, slice, dir):
            return

        i, j = (axis + 1) % 3, (axis + 2) % 3
        for k in range(3):
            self.rot[k][i], self.rot[k][j] = -self.rot[k][j] * dir, self.rot[k][i] * dir

        self.current_i[i], self.current_i[j] = (
            self.current_i[j] if dir < 0 else self.N - 1 - self.current_i[j],
            self.current_i[i] if dir > 0 else self.N - 1 - self.current_i[i])

    def transformMat(self):
        scaleA = [[s * self.scale for s in a] for a in self.rot]
        scaleT = [(p - (self.N - 1) / 2) * 2.1 * self.scale for p in self.current_i]
        return [*scaleA[0], 0, *scaleA[1], 0, *scaleA[2], 0, *scaleT, 1]

    def draw(self, col, surf, vert, animate, angle, axis, slice, dir):

        glPushMatrix()
        if animate and self.isAffected(axis, slice, dir):
            glRotatef(angle * dir, *[1 if i == axis else 0 for i in range(3)])
        glMultMatrixf(self.transformMat())

        glBegin(GL_QUADS)
        for i in range(len(surf)):
            glColor3fv(colors[i])
            for j in surf[i]:
                glVertex3fv(vertices[j])
        glEnd()

        glPopMatrix()


class EntireCube:
    def __init__(self, N, scale, shuffle_moves, solve_moves):
        """
        Create the cube

        :param N: Size of the cube
        :param scale: Scale to set the cube at
        :param shuffle_moves: Moves to shuffle the cube
        :param solve_moves: Moves to solve the cube
        """
        self.N = N
        cr = range(self.N)
        self.cubes = [Cubie((x, y, z), self.N, scale) for x in cr for y in cr for z in cr]

        self.rot_cube_map = {K_UP: (-1, 0), K_DOWN: (1, 0), K_LEFT: (0, -1), K_RIGHT: (0, 1)}
        self.rot_slice_map = {
            'L': (0, 0, 1), 'R': (0, 2, 1), 'D': (1, 0, 1),
            'U': (1, 2, 1), 'B': (2, 0, 1), 'F': (2, 2, 1),
            'L`': (0, 0, -1), 'R`': (0, 2, -1), 'D`': (1, 0, -1),
            'U`': (1, 2, -1), 'B`': (2, 0, -1), 'F`': (2, 2, -1),
        }
        self.dupe_moves = ['L2', 'R2', 'D2', 'U2', 'B2', 'F2']

        self.shuffle_moves = self.preprocess_moves(shuffle_moves)
        self.solve_moves = self.preprocess_moves(solve_moves)

    def preprocess_moves(self, moves):
        """
        Replaces double moves (i.e. 'L2') with two single moves ('L', 'L')
        :param moves: List of moves to process
        :return: List of moves with double moves replaced
        """
        for move in moves:
            if move not in self.rot_slice_map.keys() and move not in self.dupe_moves:
                raise ValueError("Invalid move given: " + str(move))

        new_moves = []

        for move in moves:
            if '2' in move:
                new_moves.extend([move[0]] * 2)
            else:
                new_moves.append(move)

        return new_moves

    def mainloop(self):
        MOVECUBE = 500  # Move cube every 1s

        ang_x, ang_y, rot_cube = 45, 45, (0, 0)
        animate, animate_ang, animate_speed = False, 0, 5
        action = (0, 0, 0)

        move_cube = pygame.USEREVENT + 1
        pygame.time.set_timer(move_cube, MOVECUBE)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == KEYDOWN:
                    if event.key in self.rot_cube_map:
                        rot_cube = self.rot_cube_map[event.key]
                if event.type == KEYUP:
                    if event.key in self.rot_cube_map:
                        rot_cube = (0, 0)
                if event.type == move_cube:
                    if len(self.shuffle_moves) > 0:
                        animate, action = True, self.rot_slice_map[self.shuffle_moves.pop(0)]
                    elif len(self.solve_moves) > 0:
                        pygame.time.set_timer(move_cube, 1000)
                        animate, action = True, self.rot_slice_map[self.solve_moves.pop(0)]

            ang_x += rot_cube[0] * 2
            ang_y += rot_cube[1] * 2

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0, 0, -40)
            glRotatef(ang_y, 0, 1, 0)
            glRotatef(ang_x, 1, 0, 0)

            if len(self.shuffle_moves) > 0:
                glClearColor(0.5, 0, 0, 0.5)
            else:
                glClearColor(0, 0.5, 0, 0.5)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            if animate:
                if animate_ang >= 90:
                    for cube in self.cubes:
                        cube.update(*action)
                    animate, animate_ang = False, 0

            for cube in self.cubes:
                cube.draw(colors, surfaces, vertices, animate, animate_ang, *action)
            if animate:
                animate_ang += animate_speed

            pygame.display.flip()
            pygame.time.wait(10)


def show_moves(shuffle_moves=None, solve_moves=None):
    if solve_moves is None:
        solve_moves = []
    if shuffle_moves is None:
        shuffle_moves = []

    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    NewEntireCube = EntireCube(3, 1.5, shuffle_moves, solve_moves)
    NewEntireCube.mainloop()
    pygame.quit()
    quit()
