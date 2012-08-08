from PySteppables import *
import CompuCell
import sys
from math import *


class MomentOfInertiaPrinter3D(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.potts=self.simulator.getPotts()        
        self.cellFieldG=self.potts.getCellFieldG()
        self.dim=self.cellFieldG.getDim()        
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)
        self.momentOfInertiaPlugin=CompuCell.getMomentOfInertiaPlugin()
        self.semiMinorAxis=15
        self.semiMedianAxis=8
        self.semiMajorAxis=5
        
        
        
        
    def start(self):
        self.generateEllipsoid(self.semiMinorAxis,self.semiMedianAxis,self.semiMajorAxis)
        
    def generateEllipsoid(self,_semiMinorAxis,_semiMedianAxis,_semiMajorAxis):        
        pt=CompuCell.Point3D(self.dim.x/2,self.dim.y/2,self.dim.z/2)
        cell=self.potts.createCellG(pt)
        cell.type=1
        
        for x in range(self.dim.x):
            for y in range(self.dim.y):
                for z in range(self.dim.z):
                    if (x-self.dim.x/2.0)**2/_semiMinorAxis**2+(y-self.dim.y/2.0)**2/_semiMajorAxis**2+(z-self.dim.z/2.0)**2/_semiMedianAxis**2 <1:
                        pt.x=x
                        pt.y=y
                        pt.z=z
                        self.cellFieldG.set(pt,cell)
                    
    def step(self,mcs):        
        for cell in self.cellList:
            #preferred way of accessing information about semiminor axes
            axes=self.momentOfInertiaPlugin.getSemiaxes(cell)            
            print "minorAxis=",axes[0]," majorAxis=",axes[2], " medianAxis=",axes[1]
            
                



