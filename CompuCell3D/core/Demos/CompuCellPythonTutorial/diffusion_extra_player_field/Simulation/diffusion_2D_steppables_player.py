from cc3d.core.PySteppables import *
from math import *


class ExtraFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.create_scalar_field_py("ExtraField")

    def step(self, mcs):
        extra_field = self.field.ExtraField

        # clearing entire field
        extra_field[:, :, :] = 0.0
        for x, y, z in self.everyPixel():
            if not mcs % 20:
                extra_field[x, y, z] = x * y
            else:
                extra_field[x, y, z] = sin(x * y)
