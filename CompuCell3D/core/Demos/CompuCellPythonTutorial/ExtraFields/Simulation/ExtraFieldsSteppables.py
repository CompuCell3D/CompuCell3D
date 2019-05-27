from cc3d.core.PySteppables import *
from math import sin
from random import random
from cc3d.cpp import CompuCell


class ExtraFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.create_scalar_field_py("ExtraField")
        self.create_scalar_field_cell_level_py("IdField")

    def step(self, mcs):

        cell = self.field.Cell_Field[20, 20, 0]
        print('cell=', cell)

        # clear field
        self.field.ExtraField[:, :, :] = 10.0

        for x, y, z in self.every_pixel(4, 4, 1):
            if not mcs % 20:
                self.field.ExtraField[x, y, z] = x * y
            # else:
            #     self.field.ExtraField[x, y, z] = sin(x * y)

        self.field.IdField.clear()
        for cell in self.cell_list:
            self.field.IdField[cell] = cell.id * random()


class IdFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # note if you create field outside constructor this field will not be properly
        # initialized if you are using restart snapshots. It is OK as long as you are aware of this limitation
        self.create_scalar_field_cell_level_py("IdFieldNew")

    def step(self, mcs):


        # clear id field
        try:
            id_field = self.field.IdFieldNew
            id_field.clear()
        except KeyError:
            # an exception might occur if you are using restart snapshots to restart simulation
            # because field has been created outside constructor
            self.create_scalar_field_cell_level_py("IdFieldNew")
            id_field = self.field.IdFieldNew

        for cell in self.cell_list:
            id_field[cell] = cell.id * random()


class VectorFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.create_vector_field_py("VectorField")

    def step(self, mcs):
        vec_field = self.field.VectorField

        # clear vector field
        vec_field[:, :, :, :] = 0.0

        for x, y, z in self.everyPixel(10, 10, 5):
            vec_field[x, y, z] = [x * random(), y * random(), z * random()]


class VectorFieldCellLevelVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

        self.create_vector_field_cell_level_py("VectorFieldCellLevel")

    def step(self, mcs):
        vec_field = self.field.VectorFieldCellLevel

        vec_field.clear()
        for cell in self.cell_list:

            if cell.type == 1:
                vec_field[cell] = [cell.id * random(), cell.id * random(), 0]
                vec = vec_field[cell]
                vec *= 2.0
                vec_field[cell] = vec


class DiffusionFieldSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # initial condition for diffusion field
        fgf_field = self.field.FGF
        fgf_field[26:28, 26:28, 0:5] = 2000.0
