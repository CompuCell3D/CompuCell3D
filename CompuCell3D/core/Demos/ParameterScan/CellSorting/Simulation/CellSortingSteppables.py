
from PySteppables import *
import CompuCell
import sys

MYVAR=10
MYVAR1='new str'

class CellSortingSteppable(SteppableBasePy):    

    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def step(self,mcs):        
        #type here the code that will run every _frequency MCS
        global MYVAR
        
        print 'MYVAR=',MYVAR
        for cell in self.cellList:
            if cell.type==self.DARK:
                # Make sure ExternalPotential plugin is loaded
                cell.lambdaVecX=-0.5 # force component pointing along X axis - towards positive X's
                
                
                
                
        