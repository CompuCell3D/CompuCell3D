from cc3d.core.PySteppables import *
from random import random


class CellsortSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        if mcs == 100:
            self.external_output = 200.0 + random()
