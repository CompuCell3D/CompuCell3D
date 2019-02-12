
from PySteppables import *
import CompuCell
import sys

class ChemotaxisCompartmentSteppable(SteppableBasePy):

    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def start(self):
        cellA = self.newCell(self.A)
        cellB = self.newCell(self.B)
        cellC = self.newCell(self.C)
        
        self.cellField[60:80, 50:60, 0] = cellA
        self.cellField[40:60, 50:60, 0] = cellB
        self.cellField[20:40, 50:60, 0] = cellC
        
        reassignIdFlag = self.inventory.reassignClusterId(cellB,1)
        reassignIdFlag = self.inventory.reassignClusterId(cellC,1)
        
        fgf = self.getConcentrationField('FGF')
        
        for x in range(20, 75,1):
            
            fgf[x,:,:] = x
        
        
        

    def step(self,mcs):        
        pass