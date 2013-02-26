from PySteppables import *
import CompuCell
import sys

class NeighborTrackerPrinterSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=100):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def step(self,mcs):
        self.cellList=CellList(self.inventory)
        for cell in self.cellList:            
            print "*********NEIGHBORS OF CELL WITH ID ",cell.id," *****************"
            for neighbor , commonSurfaceArea in self.getCellNeighborDataList(cell):                
                if neighbor:
                    print "neighbor.id",neighbor.id," commonSurfaceArea=",commonSurfaceArea
                else:
                    print "Medium commonSurfaceArea=",commonSurfaceArea


