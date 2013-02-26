
from PySteppables import *
import CompuCell
import sys

from PlayerPython import *
import CompuCellSetup
from math import *


class CellInitializer(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
            
    def start(self):        
        size=5
        xMin=15
        xMax=85
        yMin_low=15
        yMax_low=50

        yMin_high=50
        yMax_high=85


        for x in range(xMin,xMax,size):
            for y in range(yMin_low,yMax_low,size):                
                cell=self.potts.createCell()
                cell.type=self.CONDENSING
                self.cellField[x:x+size-1,y:y+size-1,0]=cell #y:y+size-1 - size -1 gets you a cell of size=size
        
        for x in range(xMin,xMax,size):
            for y in range(yMin_high,yMax_high,size):                
                cell=self.potts.createCell()
                cell.type=self.NONCONDENSING
                self.cellField[x:x+size-1,y:y+size-1,0]=cell
        
