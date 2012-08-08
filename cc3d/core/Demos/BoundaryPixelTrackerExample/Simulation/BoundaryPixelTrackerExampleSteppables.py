#Steppables

import CompuCell
import PlayerPython
from PySteppables import *


class BoundaryPixelTrackerSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def start(self):pass

    def step(self,mcs):
        for cell in self.cellList:
            pixelList=self.getCellBoundaryPixelList(cell)
            if cell.type==2:                
                for boundaryPixelTrackerData in pixelList:
                    print "pixel of cell id=",cell.id," type:",cell.type, " = ",boundaryPixelTrackerData.pixel," number of pixels=",pixelList.numberOfPixels()
