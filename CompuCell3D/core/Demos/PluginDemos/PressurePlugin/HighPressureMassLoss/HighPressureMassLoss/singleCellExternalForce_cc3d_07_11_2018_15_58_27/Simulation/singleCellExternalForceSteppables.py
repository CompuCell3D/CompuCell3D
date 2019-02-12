
from PySteppables import *
import CompuCell
import sys
class singleCellExternalForceSteppable(SteppableBasePy):

    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def start(self):
        # any code in the start function runs before MCS=0
        self.targetVolume = 36.
        self.lambdaVolume = 8.
        for cell in self.cellList():
            cell.targetVolume = self.targetVolume
            cell.lambdaVolume = self.lambdaVolume
    def step(self,mcs):        
        #type here the code that will run every _frequency MCS
        for cell in self.cellList:
            print "cell.id=",cell.id
    def finish(self):
        # Finish Function gets called after the last MCS
        pass
        