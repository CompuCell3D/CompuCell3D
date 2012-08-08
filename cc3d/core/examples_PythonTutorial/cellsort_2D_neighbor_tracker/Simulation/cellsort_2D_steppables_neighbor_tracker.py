from PySteppables import *
import CompuCell
import sys


class NeighborTrackerPrinterSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=100):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def step(self,mcs):
        self.cellList=CellList(self.inventory)
        for cell in self.cellList:
            cellNeighborList=self.getCellNeighbors(cell)
            print "*********NEIGHBORS OF CELL WITH ID ",cell.id," *****************"
            for neighborSurfaceData in cellNeighborList:
                if neighborSurfaceData.neighborAddress:
                    print "neighbor.id",neighborSurfaceData.neighborAddress.id," commonSurfaceArea=",neighborSurfaceData.commonSurfaceArea
                else:
                    print "Medium commonSurfaceArea=",neighborSurfaceData.commonSurfaceArea

