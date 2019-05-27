from cc3d.core.PySteppables import *
from random import uniform


class CellMotilitySteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        # Make sure ExternalPotential plugin is loaded
        # negative lambdaVecX makes force point in the positive direction

        for cell in self.cellList:
            # force component pointing along X axis
            cell.lambdaVecX = 10.1 * uniform(-0.5, 0.5)
            # force component pointing along Y axis
            cell.lambdaVecY = 10.1 * uniform(-0.5, 0.5)
