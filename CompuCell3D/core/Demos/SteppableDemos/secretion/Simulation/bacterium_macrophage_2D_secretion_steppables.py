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
            if cell.type==self.WALL:
                attrSecretor.secreteInsideCellAtBoundaryOnContactWith(cell,300,[self.WALL])
                attrSecretor.secreteOutsideCellAtBoundaryOnContactWith(cell,300,[self.MEDIUM])
                attrSecretor.secreteInsideCell(cell,300)
                attrSecretor.secreteInsideCellAtBoundary(cell,300)
                attrSecretor.secreteOutsideCellAtBoundary(cell,500)
                attrSecretor.secreteInsideCellAtCOM(cell,300)        

