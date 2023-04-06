
from cc3d.core.PySteppables import *

from cc3d.cpp import CompuCell

class DiffusivityFieldDemoSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """
        self.C1Diff = CompuCell.getConcentrationField(self.simulator, "C1Diff")
        self.C2Diff = CompuCell.getConcentrationField(self.simulator, "C2Diff")
        
        for x, y, z in self.everyPixel(1, 1, 1):
            if y > self.dim.y/4 and y < self.dim.y*3/4:
                if x < self.dim.x/2:
                    self.C1Diff[x, y, z] = 0.2
                    self.C2Diff[x, y, z] = 0.05
                else:
                    self.C1Diff[x, y, z] = 0.05
                    self.C2Diff[x, y, z] = 0.2
            else:
                if x < self.dim.x/2:
                    self.C1Diff[x, y, z] = 0.05
                    self.C2Diff[x, y, z] = 0.2
                else:
                    self.C1Diff[x, y, z] = 0.2
                    self.C2Diff[x, y, z] = 0.05
        

    def step(self,mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """

    def finish(self):
        """
        Finish Function is called after the last MCS
        """


        