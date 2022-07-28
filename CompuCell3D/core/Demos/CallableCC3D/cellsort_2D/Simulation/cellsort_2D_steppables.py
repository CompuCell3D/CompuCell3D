from cc3d.core.PySteppables import *
from random import random


class CellsortSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        input_val = self.external_input

    def step(self, mcs):
        if mcs == 100:
            self.external_output = 200.0 + random()
