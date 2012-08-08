from PySteppables import *
import CompuCell
import sys


class ExtraAttributeCellsort(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)

    def step(self,mcs):
        for cell in self.cellList:
            list_attrib=CompuCell.getPyAttrib(cell)
            print "length=",len(list_attrib)
            list_attrib[0:2]=[cell.id*mcs,cell.id*(mcs-1)]
            print "CELL ID modified=",list_attrib[0],"    ", list_attrib[1]


class TypeSwitcherSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=100):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)

    def step(self,mcs):
        for cell in self.cellList:
            if cell.type==1:
                cell.type=2
            elif (cell.type==2):
                cell.type=1
            else:
                print "Unknown type. In cellsort simulation there should only be two types 1 and 2"


