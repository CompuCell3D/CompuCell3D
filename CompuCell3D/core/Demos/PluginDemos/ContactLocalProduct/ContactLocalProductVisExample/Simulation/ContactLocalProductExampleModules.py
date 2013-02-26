from PlayerPython import * 
import CompuCellSetup
#Steppables

import CompuCell
from random import random
import types

from PySteppables import *

class ContactLocalProductSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def setTypeContactEnergyTable(self,_table):
        self.table=_table

    def start(self):
        for cell in self.cellList:
            specificityObj=self.table[cell.type];
            if isinstance(specificityObj,types.ListType):
                self.contactLocalProductPlugin.setJVecValue(cell,0,(specificityObj[1]-specificityObj[0])*random())
            else:
                self.contactLocalProductPlugin.setJVecValue(cell,0,specificityObj)

class ContactSpecVisualizationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)        
        self.scalarField=CompuCellSetup.createScalarFieldPy(self.dim,"ContSpec")

    def step(self,mcs):
        
        for x,y,z in self.everyPixel():
            cell=self.cellField[x,y,z]            
            if cell:
                self.scalarField[x,y,z]=self.contactLocalProductPlugin.getJVecValue(cell,0)
            else:
                self.scalarField[x,y,z]=0.0
                        
