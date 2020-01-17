from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
from random import random


class CellsortSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        pass

    def step(self, mcs):
        if mcs == 100:
            pg = CompuCellSetup.persistent_globals
            pg.return_object = 200.0 + random()
