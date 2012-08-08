#Steppables 

import CompuCell
import PlayerPython
from PySteppables import *

from PlayerPython import insertVectorIntoVectorCellLevelField as insertVector

class VectorFieldPlotTestSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.cellFieldG=self.simulator.getPotts().getCellFieldG()
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.dim=self.cellFieldG.getDim()
        
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

