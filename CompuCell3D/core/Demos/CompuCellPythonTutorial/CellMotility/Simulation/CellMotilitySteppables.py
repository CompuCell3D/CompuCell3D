from PySteppables import *
import CompuCell
import CompuCellSetup
import sys
from random import uniform

class CellMotilitySteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def start(self):
        print "This function is called once before simulation"
        
        # iterating over all cells in simulation        
        for cell in self.cellList:
            break 
            # Make sure ExternalPotential plugin is loaded
            # negative lambdaVecX makes force point in the positive direction
            cell.lambdaVecX=10.1*uniform(-0.5,0.5) # force component pointing along X axis 
            cell.lambdaVecY=10.1*uniform(-0.5,0.5) # force component pointing along Y axis 
#         cell.lambdaVecZ=0.0 # force component pointing along Z axis 
        
        
    def step(self,mcs):
        
        for cell in self.cellList:
            
            cell.lambdaVecX=10.1*uniform(-0.5,0.5) # force component pointing along X axis 
            cell.lambdaVecY=10.1*uniform(-0.5,0.5) # force component pointing along Y axis 
            # print 'cell.lambdaVecX=',cell.lambdaVecX,' cell.lambdaVecY=',cell.lambdaVecY



