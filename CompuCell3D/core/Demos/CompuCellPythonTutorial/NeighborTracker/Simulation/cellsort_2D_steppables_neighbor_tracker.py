from PySteppables import *
import CompuCell
import sys

class NeighborTrackerPrinterSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=100):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def step(self,mcs):
        
        for cell in self.cellList:            
            neighborList = self.getCellNeighborDataList(cell)
            print "*********NEIGHBORS OF CELL WITH ID ",cell.id," *****************"
            print "*********TOTAL NUMBER OF NEIGHBORS ",len(neighborList)," *****************"
            print "********* COMMON SURFACE AREA WITH TYPES (1,2) ", neighborList.commonSurfaceAreaWithCellTypes(cell_type_list=[1,2])," *****************"
            print "********* COMMON SURFACE AREA BY TYPE ", neighborList.commonSurfaceAreaByType()," *****************"
            print "********* NEIGHBOR COUNT BY TYPE ", neighborList.neighborCountByType()," *****************"
            for neighbor , commonSurfaceArea in neighborList:                
                if neighbor:
                    print "neighbor.id",neighbor.id," commonSurfaceArea=",commonSurfaceArea
                else:
                    print "Medium commonSurfaceArea=",commonSurfaceArea


