from PySteppables import *
import CompuCell
import CompuCellSetup
import sys

from PlayerPython import *
from math import *

class TargetVolumeDrosoSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def setInitialTargetVolume(self,_tv):
        self.tv=_tv
    def setInitialLambdaVolume(self,_lambdaVolume):
        self.lambdaVolume=_lambdaVolume
        
    def start(self):

        for cell in self.cellList:
            cell.targetVolume=self.tv
            cell.lambdaVolume=self.lambdaVolume
    
    def step(self,mcs):
        for cell in self.cellList:            
            if ((cell.xCOM-100)**2+(cell.yCOM-100)**2) < 400:
                cell.targetVolume+=1


class CellKiller(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def step(self,mcs):
        for cell in self.cellList:            
            if mcs==10:
                cell.targetVolume=0            


class PressureFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)                
        self.pressureField=CompuCellSetup.createScalarFieldCellLevelPy("PressureField")
        
    def step(self,mcs):
        for cell in self.cellList:
            self.pressureField[cell]=cell.targetVolume-cell.volume
            
            


