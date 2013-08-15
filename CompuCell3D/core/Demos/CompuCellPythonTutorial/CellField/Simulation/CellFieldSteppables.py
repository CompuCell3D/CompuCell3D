from PySteppables import *
import CompuCell
import sys


class UniformInitializer(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def start(self):
        print "UniformInitializer: This function is called once before simulation"
        size=5
        xMin=10
        xMax=80
        yMin=10
        yMax=80
        
        for x in range(xMin,xMax,size):
            for y in range(yMin,yMax,size):                
                self.cellField[x:x+size-1,y:y+size-1,0]=self.newCell(self.CONDENSING)
        
        cell0=self.cellField[0,0,0]
        cell1=self.cellField[50,50,0]
        print 'cells=',(cell0,cell1)
        self.cellField[50:80,50:80,0]=cell0
        
