from cc3d.core.PySteppables import *


class CellSortingSteppable(SteppableBasePy):

    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # any code in the start function runs before MCS=0
        pass

    def step(self, mcs):
        # type here the code that will run every _frequency MCS
        for cell in self.cellList:
            print("cell.id=", cell.id)

    def finish(self):
        # Finish Function gets called after the last MCS
        pass
