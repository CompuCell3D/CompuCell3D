from cc3d.core.PySteppables import *
from math import sin
from random import random
from cc3d.cpp import CompuCell


class ExtraFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.scalar_field = self.create_scalar_field_py("ExtraField")

    def start(self):

        self.scalar_cl_field = self.create_scalar_field_cell_level_py("IdField")

    def step(self, mcs):

        # clear field
        self.scalar_field[:, :, :] = 0.0

        for x, y, z in self.every_pixel(4, 4, 1):
            if (not mcs % 20):
                self.scalar_field[x, y, z] = x * y
            else:
                self.scalar_field[x, y, z] = sin(x * y)

        self.scalar_cl_field.clear()
        for cell in self.cell_list:
            self.scalar_cl_field[cell] = cell.id * random()


class IdFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        self.scalar_cl_field = self.create_scalar_field_cell_level_py("IdField")

    def step(self, mcs):
        for cell in self.cell_list:
            self.scalar_cl_field[cell] = cell.id * random()


class VectorFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.vectorField = self.create_vector_field_py("VectorField")

    def step(self, mcs):
        # clear vector field
        self.vectorField[:, :, :, :] = 0.0

        for x, y, z in self.every_pixel(10, 10, 5):
            self.vectorField[x, y, z] = [x * random(), y * random(), z * random()]


class VectorFieldCellLevelVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

        self.vectorCLField = self.create_vector_field_cell_level_py("VectorFieldCellLevel")

    def step(self, mcs):
        self.vectorCLField.clear()
        for cell in self.cell_list:

            if cell.type == 1:
                self.vectorCLField[cell] = [cell.id * random(), cell.id * random(), 0]
                vec = self.vectorCLField[cell]
                vec *= 2.0
                self.vectorCLField[cell] = vec


class DiffusionFieldSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # initial condition for diffusion field
        field = CompuCell.getConcentrationField(self.simulator, "FGF")
        field[26:28, 26:28, 0:5] = 2000.0
