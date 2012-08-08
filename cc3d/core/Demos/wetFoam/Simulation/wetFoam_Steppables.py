from PySteppables import *
import CompuCell
import sys
from random import randint
from math import *


class FlexCellInitializer(SteppableBasePy):

    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.cellTypeParameters={} 
        self.waterFraction=0.1
        
    def addCellTypeParameters(self, _type,_count,_targetVolume,_lambdaVolume):
        self.cellTypeParameters[_type]=[_count,_targetVolume,_lambdaVolume]
    def setFractionOfWater(self,_waterFraction):
        if _waterFraction>0.0 and _waterFraction<1.0:
            self.waterFraction=_waterFraction
                
    def start(self):
        self.addCellTypeParameters(_type=1,_count=0,_targetVolume=25,_lambdaVolume=10.0)
        self.addCellTypeParameters(_type=2,_count=0,_targetVolume=5,_lambdaVolume=2.0)
        
        numberOfPixels=self.dim.x*self.dim.y
        
        countType1=int(numberOfPixels*(1-self.waterFraction)/self.cellTypeParameters[1][1])
        countType2=int(numberOfPixels*self.waterFraction/self.cellTypeParameters[2][1])
        
        self.cellTypeParameters[1][0]=countType1
        self.cellTypeParameters[2][0]=countType2
        
        self.mediumCell=self.cellField.get(CompuCell.Point3D())
        pt=CompuCell.Point3D()
        for cellType,cellTypeParamList in self.cellTypeParameters.iteritems():
            #initialize self.cellTypeParameters[0]+1 number of randomly placed cells with user specified targetVolume and lambdaVolume
            for cellCount in xrange(cellTypeParamList[0]):
                pt.x=randint(0,self.dim.x-1)
                pt.y=randint(0,self.dim.y-1)
                pt.z=randint(0,self.dim.z-1)
                cell=self.potts.createCellG(pt)
                cell.type=cellType
                cell.targetVolume=cellTypeParamList[1]
                cell.lambdaVolume=cellTypeParamList[2]
    
    
    
    def step(self,mcs):
        if mcs==300:
            for cell in self.cellList:
                if cell.type==1:
                    cell.lambdaVolume=0.0
                if cell.type==2:
                    cell.lambdaVolume=10.0

    
        if mcs==200:
        
            #fill medium with water            
            pt=CompuCell.Point3D()    
            for x in xrange(self.dim.x):
                for y in xrange(self.dim.y):
                    for z in xrange(self.dim.z):
                    
                        pt.x=x
                        pt.y=y
                        pt.z=z
                        currentCell=self.cellField.get(pt)
                        if not currentCell:
                            cell=self.potts.createCellG(pt)
                            cell.type=2
                            
                            cell.targetVolume=self.cellTypeParameters[2][1]
                            cell.lambdaVolume=self.cellTypeParameters[2][2]
                    
                    
            
        
        
