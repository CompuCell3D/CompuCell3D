from PySteppables import *
import CompuCell
import sys
import math

class GrowthSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def step(self,mcs):
        for cell in self.cellList:
            cell.targetVolume+=1             

class OrientedConstraintSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency,_OGPlugin):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.OGPlugin = _OGPlugin
        
    def start(self):
        for cell in self.cellList:
            cell.lambdaVolume=2.0
            cell.targetVolume=cell.volume
            
            self.OGPlugin.setElongationAxis(cell, math.cos(math.pi / 3), math.sin(math.pi / 3)) # Here, we define the axis of elongatino.
            self.OGPlugin.setConstraintWidth(cell, 2.0) # And this function gives a 2 pixel width to each cell
            self.OGPlugin.setElongationEnabled(cell, True) # Make sure to enable or disable elongation in all cells
                                                            # Or unexpected results may occur.