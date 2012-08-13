from PySteppables import *
import CompuCell
import sys


from PlayerPython import *
from math import *

class FieldSecretionSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.cellFieldG=self.simulator.getPotts().getCellFieldG()
        self.dim=self.cellFieldG.getDim()
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)

    def setScalarFieldName(self,_fieldName):
        self.fieldName=_fieldName
        self.scalarField=CompuCell.getConcentrationField(self.simulator,self.fieldName)
        assert self.scalarField, "you have picked field: "+self.fieldName+" which does not exist"


    def start(self):pass

    def step(self,mcs):
        lmfLength=1.0;
        xScale=1.0
        yScale=1.0
        zScale=1.0
        # FOR HEX LATTICE IN 2D
#         lmfLength=sqrt(2.0/(3.0*sqrt(3.0)))*sqrt(3.0)
#         xScale=1.0
#         yScale=sqrt(3.0)/2.0
#         zScale=sqrt(6.0)/3.0

        for cell in self.cellList:
            xCM=int(cell.xCM/float(cell.volume*lmfLength*xScale))
            yCM=int(cell.yCM/float(cell.volume*lmfLength*yScale))
            pt=CompuCell.Point3D(xCM,yCM,0)
            if cell.type==1:
                self.scalarField.set(pt,10)


            elif cell.type==2:
                self.scalarField.set(pt,20)


