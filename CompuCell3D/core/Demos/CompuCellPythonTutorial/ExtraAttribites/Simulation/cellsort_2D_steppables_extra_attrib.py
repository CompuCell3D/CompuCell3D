from PySteppables import *
import CompuCell
import sys


class ExtraAttributeCellsort(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def step(self,mcs):
        for cell in self.cellList:
            list_attrib=CompuCell.getPyAttrib(cell)
            print "length=",len(list_attrib)
            list_attrib[0:2]=[cell.id*mcs,cell.id*(mcs-1)]
            print "CELL ID modified=",list_attrib[0],"    ", list_attrib[1]


class TypeSwitcherSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=100):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def step(self,mcs):
        for cell in self.cellList:
            if cell.type==self.CONDENSING:
                cell.type=self.NONCONDENSING
            elif cell.type==self.NONCONDENSING:
                cell.type=self.CONDENSING
            else:
                print "Unknown type. In cellsort simulation there should only be two types 1 and 2"


