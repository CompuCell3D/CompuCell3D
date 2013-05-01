
from PySteppables import *
import CompuCell
import sys
import numpy

class CellDistanceSteppable(SteppableBasePy):    

    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.cellA=None
        self.cellB=None
    def start(self):
        self.cellA=self.potts.createCell()    
        self.cellA.type=self.A
        self.cellField[10:12,10:12,0]=self.cellA
        
        self.cellB=self.potts.createCell()    
        self.cellB.type=self.B
        self.cellField[92:94,10:12,0]=self.cellB
        
    def step(self,mcs):  


        distVec=self.invariantDistanceVectorInteger(_from=[10,10,0] ,_to=[92,12,0])
        print 'distVec=',distVec, ' norm=',self.vectorNorm(distVec)
        
        distVec=self.invariantDistanceVector(_from=[10,10,0] ,_to=[92.3,12.1,0])
        print 'distVec=',distVec, ' norm=',self.vectorNorm(distVec)
        
        print 'distance invariant=',self.invariantDistance(_from=[10,10,0] ,_to=[92.3,12.1,0])
        
        print 'distance =',self.distance(_from=[10,10,0] ,_to=[92.3,12.1,0])
        
        print 'distance vector between cells =',self.distanceVectorBetweenCells(self.cellA,self.cellB)
        print 'invariant distance vector between cells =',self.invariantDistanceVectorBetweenCells(self.cellA,self.cellB)
        print 'distanceBetweenCells = ',self.distanceBetweenCells(self.cellA,self.cellB)
        print 'invariantDistanceBetweenCells = ',self.invariantDistanceBetweenCells(self.cellA,self.cellB)
        
        