#Steppables

import CompuCell
import PlayerPython
from PySteppables import *


class BoundaryMonitorSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def start(self):
        self.pixelAssigned=False
        self.mediumCell=CompuCell.getMediumCell()   
        self.boundaryArray=self.boundaryMonitorPlugin.getBoundaryArray()
        print 'self.boundaryArray=',self.boundaryArray
        print 'dir(self.boundaryArray)=',dir(self.boundaryArray)
        
        
    def step(self,mcs):
        for cell in self.cellList:            
            if cell.type==3:                
                    
                pt=CompuCell.Point3D()    
                for x in range (9,17):
                    for y in range (9,17):
                        pt.x=x
                        pt.y=y
                        if int(self.boundaryArray.get(pt)):
                                print 'pt=',pt,' boundary=',int(self.boundaryArray.get(pt))
            if not self.pixelAssigned:
                 
                 pt=CompuCell.Point3D(12,12,0)
                 self.cellField.set(pt,self.mediumCell)
                 self.pixelAssigned=True
            if mcs==3:
                pt=CompuCell.Point3D(12,12,0)
                self.cellField.set(pt,cell)
                print 'REASSIGNMNET COMPLETED'

            if mcs==4:
                pt=CompuCell.Point3D(12 ,10,0)
                self.cellField.set(pt,self.mediumCell)
            if mcs==5:
                pt=CompuCell.Point3D(12 ,11,0)
                self.cellField.set(pt,self.mediumCell)
                
                # print 'REASSIGNMNET COMPLETED'
                
            break
             