from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
import numpy as np
from scipy import special, optimize
from scipy import integrate
from numpy import exp
from scipy import stats


class FieldSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency=frequency)

    def start(self):
        """

        :return:
        """

        self.field.F[-4:, -4:, :] = 2.0
        self.field.F[0:4, 2:4, :] = 2.0
        self.field.F[-3, -3, :] = 5.0
        print(self.field.F[19, 0, 0])
        print(self.field.F[20, 0, 0])
        # for x, y, z in self.every_pixel():
        #     print(f'{x}, {y}, {z} = ', self.field.F[x, y, z])

        cell = self.new_cell(self.AMOEBA)
        self.cell_field[:4, :4, :] = cell
        self.cell_field[-4:, -4:, :] = cell

        self.cell_field[::2, ::3, :] = cell

        # size = 5
        # x_min = 10
        # x_max = 50
        # y_min = 10
        # y_max = 50
        #
        # for x in range(x_min, x_max, size):
        #     for y in range(y_min, y_max, size):
        #         # print
        #         self.cell_field[x:x + size - 1, y:y + size - 1, 0] = self.new_cell(self.AMOEBA)

        # cell0 = self.cell_field[0, 0, 0]
        # cell1 = self.cell_field[50, 50, 0]
        # print('cells=', (cell0, cell1))
        # self.cell_field[50:80, 50:80, 0] = cell0


    def step(self, mcs):
        """

        :param mcs:
        :return:
        """
