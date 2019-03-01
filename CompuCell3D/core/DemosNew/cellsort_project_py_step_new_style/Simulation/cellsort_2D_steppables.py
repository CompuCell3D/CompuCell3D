from cc3d.core.PySteppables import *
# from cc3d import CompuCellSetup
import sys
import time

class CellsortSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency=frequency)
            
    def start(self):        
        print("INSIDE START FUNCTION")

    def step(self, mcs):
        print("running mcs=", mcs)
        for i, cell in enumerate(self.cellList):
            if i > 3:
                break
            # print ('cell=', cell)
            print ('cell.id=', cell.id)

        print('sleeping')
        time.sleep(0.3)

        print('woke up')

        if mcs ==50:
            # CompuCellSetup.stop_simulation()
            self.stop_simulation()
        
        #     print ('cell.type=', cell.type)
        #     print ('cell.volume=', cell.volume)

        # for cell in self.cellListByType(1):
        #     print (' BY TYPE cell=', cell, ' cell.id=', cell.id, 'cell.type=', cell.type)

        
        # for compartmentList in self.clusterList:
        #     print ("cluster has size=",compartmentList.size())
        #     clusterId=0
        #     clusterVolume=0            
        #     for cell in CompartmentList(compartmentList):
        #         print("compartment.id=",cell.id)


