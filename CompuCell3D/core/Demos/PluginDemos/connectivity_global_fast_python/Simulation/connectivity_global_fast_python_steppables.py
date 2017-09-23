
from PySteppables import *
import CompuCell
import sys

from PlayerPython import *
import CompuCellSetup
from math import *


class ConnectivitySteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def start(self):
        # will tunr connectivity for first 100 cells
        for cell in self.cellList:
            if cell.id < 100:
                cell.connectivityOn = True                    
        
    def step(self,mcs):
         pass
#         for cell in self.cellList:            
#             print 'cell.connectivityOn =', cell.connectivityOn 
        
        

        
    
