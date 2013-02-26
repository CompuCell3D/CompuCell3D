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


