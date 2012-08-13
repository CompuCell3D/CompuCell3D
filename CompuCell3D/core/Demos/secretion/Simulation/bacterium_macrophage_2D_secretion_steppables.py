from PySteppables import *
import CompuCell
import sys
from XMLUtils import dictionaryToMapStrStr as d2mss
            
class SecretionSteppable(SecretionBasePy):
    def __init__(self,_simulator,_frequency=1):
        SecretionBasePy.__init__(self,_simulator, _frequency)
        
    def step(self,mcs):
        attrSecretor=self.getFieldSecretor("ATTR")
        for cell in self.cellList:
            if cell.type==3:
                attrSecretor.secreteInsideCell(cell,300)
                attrSecretor.secreteInsideCellAtBoundary(cell,300)
                attrSecretor.secreteOutsideCellAtBoundary(cell,500)
                attrSecretor.secreteInsideCellAtCOM(cell,300)        

