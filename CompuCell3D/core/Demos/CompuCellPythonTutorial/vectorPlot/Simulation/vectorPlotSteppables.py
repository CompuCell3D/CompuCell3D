#Steppables 

import CompuCell
import PlayerPython
from PySteppables import *

from PlayerPython import insertVectorIntoVectorCellLevelField as insertVector

class VectorFieldPlotTestSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator, _frequency)
        
    def setVectorField(self,_field):
        self.vectorField=_field
    def start(self):pass

    def step(self,mcs):
        invItr=CompuCell.STLPyIteratorCINV()
        invItr.initialize(self.inventory.getContainer())
        invItr.setToBegin()
        cell=invItr.getCurrentRef()
        offset=10
        PlayerPython.clearVectorCellLevelField(self.vectorField)
        while (1):
            if invItr.isEnd():
                break
            cell=invItr.getCurrentRef()
            insertVector(self.vectorField,cell,mcs+offset+1,mcs+offset+2,mcs+offset+3)
            offset+=10
            invItr.next()

