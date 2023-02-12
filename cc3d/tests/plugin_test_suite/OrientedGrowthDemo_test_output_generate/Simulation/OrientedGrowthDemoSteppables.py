from cc3d.core.PySteppables import *
import math


class GrowthSteppable(SteppableBasePy):
    def __init__(self, frequency):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        for cell in self.cell_list:
            cell.targetVolume += 1


class OrientedConstraintSteppable(SteppableBasePy):
    def __init__(self, frequency):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        for cell in self.cellList:
            cell.lambdaVolume = 2.0
            cell.targetVolume = cell.volume

            # Here, we define the axis of elongatino.
            self.orientedGrowthPlugin.setElongationAxis(cell, math.cos(math.pi / 3),
                                                        math.sin(math.pi / 3))
            # And this function gives a 2 pixel width to each cell
            self.orientedGrowthPlugin.setConstraintWidth(cell, 2.0)

            # Make sure to enable or disable elongation in all cells
            self.orientedGrowthPlugin.setElongationEnabled(cell, True)
            # Or unexpected results may occur.
