from cc3d.core.PySteppables import *
import sys
import time


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
            # print ('cell=', cell)
            print('cell.id=', cell.id)

        print('sleeping')
        time.sleep(0.3)

        print('woke up')

        if mcs == 50:
            self.stop_simulation()
