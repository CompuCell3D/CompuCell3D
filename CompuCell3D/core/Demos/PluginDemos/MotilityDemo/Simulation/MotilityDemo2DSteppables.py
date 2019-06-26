from cc3d.core.PySteppables import *


class MotilityDemo2DSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # any code in the start function runs before MCS=0
        pass

    def step(self, mcs):
        for cell in self.cellList:
            print("\n cell.biasVecX=", cell.biasVecX)
            print("\n cell.biasVecY=", cell.biasVecX)
            print("\n cell.biasVecZ=", cell.biasVecZ)
        pass

    def finish(self):
        # Finish Function gets called after the last MCS
        pass
