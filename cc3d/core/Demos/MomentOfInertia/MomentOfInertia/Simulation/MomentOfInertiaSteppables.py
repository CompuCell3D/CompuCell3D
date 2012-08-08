from PySteppables import *
import CompuCell
import sys
from math import *


class MomentOfInertiaPrinter(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.semiMinorAxis=15
        self.semiMajorAxis=5
        
        
        
    def start(self):
        self.generateEllipse(self.semiMinorAxis,self.semiMajorAxis)
        
    def generateEllipse(self,_semiMinorAxis,_semiMajorAxis):        
        pt=CompuCell.Point3D(self.dim.x/2,self.dim.y/2,0)
        cell=self.potts.createCellG(pt)
        cell.type=1
        
        for x in range(self.dim.x):
            for y in range(self.dim.y):
                if (x-self.dim.x/2.0)**2/_semiMinorAxis**2+(y-self.dim.y/2.0)**2/_semiMajorAxis**2<1:
                    pt.x=x
                    pt.y=y
                    self.cellField.set(pt,cell)
                    
    def step(self,mcs):        
        for cell in self.cellList:
            print "CELL ID=",cell.id, " CELL TYPE=",cell.type," volume=",cell.volume , " lX=",cell.lX," lY=",cell.lY, " ecc=",cell.ecc
            print "cell.iXX=",cell.iXX," cell.iYY=",cell.iYY," cell.iXY=",cell.iXY
            radical=0.5*sqrt((cell.iXX-cell.iYY)*(cell.iXX-cell.iYY)+4.0*cell.iXY*cell.iXY)	
            print "radical=",radical
            lMin=0.5*(cell.iXX+cell.iYY)-radical
            lMax=0.5*(cell.iXX+cell.iYY)+radical            
            # b=sqrt(cell.iXY**2+(lMax-cell.iXX)**2)
            # print "|b|=",b
            lSemiMajor=2*sqrt(lMax/cell.volume)
            lSemiMinor=2*sqrt(lMin/cell.volume)
            ecc=sqrt(1-(lSemiMinor/lSemiMajor)**2)
            print "lSemiMajor=",lSemiMajor," lSemiMinor=",lSemiMinor
            print "cell.ecc=",cell.ecc," ecc=",ecc
            print "lMin=",lMin/cell.volume," lMax=",lMax/cell.volume
            
            #preferred way of accessing information about semiminor axes
            axes=self.momentOfInertiaPlugin.getSemiaxes(cell)            
            print "minorAxis=",axes[0]," majorAxis=",axes[2], " medianAxis=",axes[1]



