from PySteppables import *
import CompuCell
import CompuCellSetup
import sys
from PlayerPython import *
from math import *

class ExtraFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.scalarField=CompuCellSetup.createScalarFieldPy(self.dim,"ExtraField")
        
    def step(self,mcs):
        self.scalarField[:,:,:]=0.0 #clearing entire field
        for x,y,z in self.everyPixel():            
            if (not mcs%20):                
                self.scalarField[x,y,z]=x*y                
            else:
                self.scalarField[x,y,z]=sin(x*y)

