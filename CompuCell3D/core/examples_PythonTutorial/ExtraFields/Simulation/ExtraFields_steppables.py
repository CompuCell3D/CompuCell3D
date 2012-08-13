from PySteppables import *
import CompuCell
import sys


from PlayerPython import *
from math import *

class ExtraFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def setScalarField(self,_field):
        self.scalarField=_field

    def start(self):pass

    def step(self,mcs):
        clearScalarField(self.dim,self.scalarField)
        for x in xrange(self.dim.x):
            for y in xrange(self.dim.y):
                for z in xrange(self.dim.z):
                    pt=CompuCell.Point3D(x,y,z)
                    if (not mcs%20):
                        value=x*y
                        fillScalarValue(self.scalarField,x,y,z,value)
                    else:
                        value=sin(x*y)
                        fillScalarValue(self.scalarField,x,y,z,value)

class IdFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def setScalarField(self,_field):
        self.scalarField=_field

    def step(self,mcs):
        clearScalarValueCellLevel(self.scalarField)
        from random import random
        for cell in self.cellList:

            fillScalarValueCellLevel(self.scalarField,cell,cell.id*random())

class VectorFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def setVectorField(self,_field):
        self.vectorField=_field
    
    def step(self,mcs):
        maxLength=0
        clearVectorField(self.dim,self.vectorField)
        import math
        for x in xrange(0,self.dim.x,5):
            for y in xrange(0,self.dim.y,5):
                for z in xrange(self.dim.z):
                     
                    pt=CompuCell.Point3D(x,y,z)
                    
                    insertVectorIntoVectorField(self.vectorField,pt.x, pt.y,pt.z, pt.x, pt.y, pt.z)

class VectorFieldCellLevelVisualizationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def setVectorField(self,_field):
        self.vectorField=_field

    def step(self,mcs):
        clearVectorCellLevelField(self.vectorField)
        for cell in self.cellList:
            if cell.type==1:
                insertVectorIntoVectorCellLevelField(self.vectorField,cell, cell.id, cell.id, 0.0)
            
        