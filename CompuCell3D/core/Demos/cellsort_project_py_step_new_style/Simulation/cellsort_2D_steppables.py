from cc3d.core.PySteppables import *


class CellsortSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency=frequency)

    def start(self):
        print("INSIDE START FUNCTION")

    def step(self, mcs):
        print("running mcs=", mcs)
        for i, cell in enumerate(self.cell_list):
            if i > 3:
                break
            print('cell.id=', cell.id)
