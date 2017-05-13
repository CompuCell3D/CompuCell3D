from PySteppables import *
import CompuCell
import sys


class ExtraAttributeCellsort(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def step(self,mcs):
        for cell in self.cellList:
            
            cell.dict['my_list']=[cell.id*mcs,cell.id*(mcs-1)]
            print "CELL ID modified=",cell.dict['my_list'][0],"    ", cell.dict['my_list'][1]


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


