from PySteppables import *
import CompuCell
import sys


from PlayerPython import *
from math import *

class FieldSecretionSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def step(self,mcs):
        self.scalarField=self.getConcentrationField('FGF')
        
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
            #converting from real coordinates to pixels
            xCM=int(cell.xCOM/(lmfLength*xScale))            
            yCM=int(cell.yCOM/(lmfLength*yScale))
            
            if cell.type==self.AMOEBA:
                self.scalarField[xCM,yCM,0]=10.0
                
            elif cell.type==self.BACTERIA:
                self.scalarField[xCM,yCM,0]=20.0
                


