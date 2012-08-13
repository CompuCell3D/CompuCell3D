from PySteppables import *
import CompuCell
import sys

class TargetVolumeDrosoSteppable(SteppablePy):
    def __init__(self,_frequency=1):
        SteppablePy.__init__(self,_frequency)
        self.inventory=None
    #def __name__(self):
        #self.name
    def setInitialTargetVolume(self,_tv):
        self.tv=_tv
    def setInitialLambdaVolume(self,_lambdaVolume):
        self.lambdaVolume=_lambdaVolume
    def init(self,_simulator):
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)
        
    def start(self):

        for cell in self.cellList:
            print "CELL ID=",cell.id
            cell.targetVolume=self.tv
            cell.lambdaVolume=self.lambdaVolume
    
    def step(self,mcs):
        xCM=0
        yCM=0
        yCM=0
        
        for cell in self.cellList:
            print "Cell.id=",cell.id," volume=",cell.volume
            xCM=cell.xCM/float(cell.volume)
            yCM=cell.yCM/float(cell.volume)
            zCM=cell.zCM/float(cell.volume)
            if ((xCM-100)**2+(yCM-100)**2) < 400:
                cell.targetVolume+=1

class BlobSimpleTypeInitializer(SteppablePy):
    def __init__(self,_simulator,_frequency=1):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)
    def start(self):
        for cell in self.cellList:
            print "Blob Initializer CELL ID=",cell.id
            cell.type=1

class ModifyAttribute(SteppablePy):
    def __init__(self,_simulator,_frequency=1):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)
    def start(self):
        for cell in self.cellList:
            print " MODIFY ATTRIB CELL ID=",cell.id
            #print "ref count=",sys.getrefcount(cell.pyAttrib)
            #print "another ref count=",sys.getrefcount(cell.pyAttrib)
            #pyAttrib=cell.pyAttrib
            pyAttrib=CompuCell.getPyAttrib(cell)
            print "ref count=",sys.getrefcount(pyAttrib)
            #print " after assign ref count=",sys.getrefcount(cell.pyAttrib)
            
            print "length=",len(pyAttrib)
            pyAttrib[0:1]=[cell.id*4]
            print "Cell attrib=",pyAttrib[0]
            print "length=",len(pyAttrib)
            pyAttrib.append(14)
            print "Cell attrib=",pyAttrib[1]
            print "length=",len(pyAttrib)
            
            #accessing cell.pyAttrib directly causes ref count decrease and objects might be garbage collected prematurely.
            #always use CompuCell.getPyAttrib(cell) function call. 
            #print "length=",len(cell.pyAttrib)
            #cell.pyAttrib=4
            #list_temp=cell.pyAttrib
            #cell.pyAttrib[0]=11
            #cell.pyAttrib[0]=12
            #cell.pyAttrib.append(cell.id*5)
            #cell.pyAttrib.append(cell.id*5)
            #print "After length=",len(cell.pyAttrib)
            #print "Cell attrib=",cell.pyAttrib[0]


    def step(self,mcs):
        for cell in self.cellList:
            #pyAttrib=cell.pyAttrib
            pyAttrib=CompuCell.getPyAttrib(cell)
            pyAttrib[0]=[cell.id*mcs,cell.id*(mcs-1)]
            if not mcs % 20:
                print "CELL ID modified=",pyAttrib," true=",cell.id,
                print " ref count=",sys.getrefcount(pyAttrib),
                print " ref count=",sys.getrefcount(pyAttrib)


class CellKiller(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)

    def step(self,mcs):
        for cell in self.cellList:
            print "Step cell.targetVolume=",cell.targetVolume
            if mcs==10:
                cell.targetVolume=0
                print "cell.targetVolume=",cell.targetVolume

from PlayerPython import *
from math import *

class PressureFieldVisualizationSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.cellFieldG=self.simulator.getPotts().getCellFieldG()
        self.dim=self.cellFieldG.getDim()

    def setScalarField(self,_field):
        self.scalarField=_field

    def start(self):pass

    def step(self,mcs):
        for x in xrange(self.dim.x):
            for y in xrange(self.dim.y):
                for z in xrange(self.dim.z):
                    pt=CompuCell.Point3D(x,y,z)
                    cell=self.cellFieldG.get(pt)
                    if cell:
                        fillScalarValue(self.scalarField,x,y,z, cell.targetVolume-cell.volume)
                    else:
                        fillScalarValue(self.scalarField,x,y,z,0)


