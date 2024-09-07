from cc3d.core.PySteppables import *


class CellSortingSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)


    def step(self, mcs):
        """
        Code in this section runs every MCS or every frequency-MCS
        """
