from cc3d.core.PySteppables import *
from math import sin
from random import random


class ExtraFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.scalar_field = self.create_scalar_field_py("ExtraField")

    def start(self):
        # self.scalar_field = self.create_scalar_field_py("ExtraField")

        print('inside start')

        self.scalar_cl_field = self.create_scalar_field_cell_level_py("IdField")

    def step(self, mcs):
        print
        self.scalar_field[:, :, :] = 0.0  # clear field

        for x, y, z in self.every_pixel(4, 4, 1):
            if (not mcs % 20):
                self.scalar_field[x, y, z] = x * y
            else:
                self.scalar_field[x, y, z] = sin(x * y)

        for cell in self.cellList:
            self.scalar_cl_field[cell] = cell.id * random()


class IdFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # self.scalar_field = self.create_scalar_field_py("ExtraField")

        print('inside start')

        self.scalar_cl_field = self.create_scalar_field_cell_level_py("IdField")

    def step(self, mcs):
        for cell in self.cellList:
            self.scalar_cl_field[cell] = cell.id * random()


class VectorFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.vectorField = self.create_vector_field_py("VectorField")

    def step(self, mcs):
        self.vectorField[:, :, :, :] = 0.0  # clear vector field

        for x, y, z in self.everyPixel(10, 10, 5):
            self.vectorField[x, y, z] = [x * random(), y * random(), z * random()]


class VectorFieldCellLevelVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

        self.vectorCLField = self.create_vector_field_cell_level_py("VectorFieldCellLevel")

    def step(self, mcs):
        # todo 5 - fix it
        self.vectorCLField.clear()
        for cell in self.cellList:

            if cell.type == 1:
                self.vectorCLField[cell] = [cell.id * random(), cell.id * random(), 0]
                vec = self.vectorCLField[cell]
                vec *= 2.0
                self.vectorCLField[cell] = vec

# import CompuCellSetup
# from PySteppables import *
# import CompuCell
# import sys


# from PlayerPython import *
# from math import *
# from random import random


# class IdFieldVisualizationSteppable(SteppableBasePy):
#     def __init__(self,_simulator,_frequency=10):
#         SteppableBasePy.__init__(self,_simulator,_frequency)        
#         self.scalarCLField=self.createScalarFieldCellLevelPy("IdField")

#     def step(self,mcs):
#         self.scalarCLField.clear()

#         for cell in self.cellList:

#             self.scalarCLField[cell]=cell.id*random()


# class VectorFieldVisualizationSteppable(SteppableBasePy):
#     def __init__(self,_simulator,_frequency=10):
#         SteppableBasePy.__init__(self,_simulator,_frequency)        
#         self.vectorField=self.createVectorFieldPy("VectorField")

#     def step(self,mcs):

#         self.vectorField[:, :, :,:]=0.0 # clear vector field        

#         for x,y,z in self.everyPixel(10,10,5):
#             self.vectorField[x,y,z]=[x*random(), y*random(), z*random()]


# class VectorFieldCellLevelVisualizationSteppable(SteppableBasePy):
#     def __init__(self,_simulator,_frequency=10):
#         SteppableBasePy.__init__(self,_simulator,_frequency)     

#         self.vectorCLField=self.createVectorFieldCellLevelPy("VectorFieldCellLevel")        

#     def step(self,mcs):
#         self.vectorCLField.clear()
#         for cell in self.cellList:

#             if cell.type==1:                

#                 self.vectorCLField[cell]=[cell.id*random(),cell.id*random(),0]
#                 vec=self.vectorCLField[cell]
#                 vec*=2.0
#                 self.vectorCLField[cell]=vec


# class DiffusionFieldSteppable(SteppableBasePy):
#     def __init__(self,_simulator,_frequency=1):
#         SteppableBasePy.__init__(self,_simulator,_frequency)

#     def start(self):  
#         #initial condition for diffusion field    
#         field=CompuCell.getConcentrationField(self.simulator,"FGF")        
#         field[26:28,26:28,0:5]=2000.0
