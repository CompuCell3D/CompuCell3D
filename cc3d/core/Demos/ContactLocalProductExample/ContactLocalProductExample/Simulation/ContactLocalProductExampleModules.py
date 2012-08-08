#Steppables

"This module contains examples of certain more and less useful steppables written in Python"
from CompuCell import NeighborFinderParams
import CompuCell
from random import random
import types

from PySteppables import *

class ContactLocalProductSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.contactProductPlugin=CompuCell.getContactLocalProductPlugin()
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)
    def setTypeContactEnergyTable(self,_table):
        self.table=_table

    def start(self):
        for cell in self.cellList:
            specificityObj=self.table[cell.type];
            if isinstance(specificityObj,types.ListType):
                self.contactProductPlugin.setJVecValue(cell,0,(specificityObj[1]-specificityObj[0])*random())
            else:
                self.contactProductPlugin.setJVecValue(cell,0,specificityObj)


from PlayerPython import fillScalarValue as conSpecSet
class ContactSpecVisualizationSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.contactProductPlugin=CompuCell.getContactLocalProductPlugin()
        self.cellFieldG=self.simulator.getPotts().getCellFieldG()
        self.dim=self.cellFieldG.getDim()
        
    def setScalarField(self,_field):
        self.scalarField=_field
    def start(self):pass

    def step(self,mcs):
        cell=None
        cellFieldG=self.cellFieldG
        for x in xrange(self.dim.x):
            for y in xrange(self.dim.y):
                for z in xrange(self.dim.z):
                    pt=CompuCell.Point3D(x,y,z)
                    cell=cellFieldG.get(pt)
                    if cell:
                        conSpecSet(self.scalarField,x,y,z,self.contactProductPlugin.getJVecValue(cell,0))
                    else:
                        conSpecSet(self.scalarField,x,y,z,0.0)
