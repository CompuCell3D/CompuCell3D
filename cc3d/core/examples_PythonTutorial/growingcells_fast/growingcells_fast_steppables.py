from PySteppables import *
import CompuCell
import sys


class VolumeParamSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=1):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)
    def start(self):
        for cell in self.cellList:
            cell.targetVolume=25
            cell.lambdaVolume=2.0
    def step(self,mcs):
        for cell in self.cellList:
            cell.targetVolume+=1

