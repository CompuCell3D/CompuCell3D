from cc3d.core.PySteppables import *


class ExtraAttributeCellsort(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        for cell in self.cell_list:
            print('demo')
            cell.dict['my_list'] = [cell.id * mcs, cell.id * (mcs - 1)]
            print("CELL ID modified=", cell.dict['my_list'][0], "    ", cell.dict['my_list'][1])


class TypeSwitcherSteppable(SteppableBasePy):
    def __init__(self, frequency=100):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        for cell in self.cell_list:
            if cell.type == self.CONDENSING:
                cell.type = self.NONCONDENSING
            elif cell.type == self.NONCONDENSING:
                cell.type = self.CONDENSING
            else:
                print("Unknown type. In cellsort simulation there should only be two types 1 and 2")
