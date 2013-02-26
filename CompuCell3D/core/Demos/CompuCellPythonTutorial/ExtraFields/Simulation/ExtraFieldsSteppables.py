import CompuCellSetup
from PySteppables import *
import CompuCell
import sys


from PlayerPython import *
from math import *
from random import random

class ExtraFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.scalarField=CompuCellSetup.createScalarFieldPy(self.dim,"ExtraField")
    def step(self,mcs):
        
        self.scalarField[:, :, :]=0.0 # clear field
        
        for x in xrange(self.dim.x):
            for y in xrange(self.dim.y):
                for z in xrange(self.dim.z):
                    
                    if (not mcs%20):
                        self.scalarField[x,y,z]=x*y
                        
                    else:
                        self.scalarField[x,y,z]=sin(x*y)

class IdFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)        
        self.scalarCLField=CompuCellSetup.createScalarFieldCellLevelPy("IdField")
        
    def step(self,mcs):
        self.scalarCLField.clear()
        
        for cell in self.cellList:
            
            self.scalarCLField[cell]=cell.id*random()
            


class VectorFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)        
        self.vectorField=CompuCellSetup.createVectorFieldPy(self.dim,"VectorField")
    
    def step(self,mcs):
        
        self.vectorField[:, :, :,:]=0.0 # clear vector field        
        
        for x in xrange(0,self.dim.x,5):
            for y in xrange(0,self.dim.y,5):
                for z in xrange(self.dim.z):
                    
                    self.vectorField[x,y,z]=[x*random(), y*random(), z*random()]

class VectorFieldCellLevelVisualizationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)     
        
        self.vectorCLField=CompuCellSetup.createVectorFieldCellLevelPy("VectorFieldCellLevel")        

    def step(self,mcs):
        self.vectorCLField.clear()
        for cell in self.cellList:
            
            if cell.type==1:                
                
                self.vectorCLField[cell]=[cell.id*random(),cell.id*random(),0]
                vec=self.vectorCLField[cell]
                vec*=2.0
                self.vectorCLField[cell]=vec
            

class DiffusionFieldSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def start(self):  
        #initial condition for diffusion field    
        field=CompuCell.getConcentrationField(self.simulator,"FGF")        
        field[26:28,26:28,0:5]=2000.0
        
