from cc3d.core.PySteppables import *


# import CompuCell
# import sys
#
# from PlayerPython import *
# from math import *
# from random import random


class DiffusionFieldSteppable(SecretionBasePy):
    # IMPORTANT MAKE SURE YOU INHERIT FROM SecretionBasePy when you use
    # steadyState solver and manage secretion from Python

    def __init__(self, frequency=1):
        SecretionBasePy.__init__(self, frequency)

    def start(self):
        # initial condition for diffusion field
        self.field = CompuCell.getConcentrationField(self.simulator, "FGF")

        # a bit slow code - one can optimize it in the production runs
        secr_const = 10
        for x, y, z in self.every_pixel(1, 1, 1):
            cell = self.cell_field[x, y, z]
            if cell and cell.type == 1:
                # notice for steady state solver we do not add secretion const to existing concentration
                # Also notice that secretion has to be negative (if we want positive secretion).
                # This is how the solver is coded
                self.field[x, y, z] = -secr_const
            else:
                # for steady state solver all field pixels which do not secrete or uptake must me set to 0.0.
                # This is how the solver works:
                # non-zero value of the field at the pixel indicates secretion rate
                self.field[x, y, z] = 0.0

    def step(self, mcs):

        # a bit slow - will write faster version
        secr_const = mcs
        for x, y, z in self.every_pixel(1, 1, 1):
            cell = self.cell_field[x, y, z]
            if cell and cell.type == 1:
                # notice for steady state solver we do not add secretion const to existing concentration
                # Also notice that secretion has to be negative (if we want positive secretion).
                # This is how the solver is coded
                self.field[x, y, z] = -secr_const
            else:
                # for steady state solver all field pixels which do not secrete or uptake must me set to 0.0.
                # This is how the solver works:
                # non-zero value of the field at the pixel indicates secretion rate
                self.field[x, y, z] = 0.0
