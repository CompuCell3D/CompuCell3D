
from cc3d.core.PySteppables import *

from cc3d.CompuCellSetup import persistent_globals as pg
from cc3d.cpp import CompuCell
import random

class SchnakenbergModelSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """

    def step(self,mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """
        if mcs == 0:
            fieldC1 = CompuCell.getConcentrationField(pg.simulator, "C1")
            fieldC2 = CompuCell.getConcentrationField(pg.simulator, "C2")
            
            for x, y, z in self.every_pixel():
                fieldC1[x, y, z] = random.random()
            
            for cell in [cell for cell in self.cell_list if cell]:
                for pt in self.get_cell_pixel_list(cell):
                    fieldC1[pt.pixel.x, pt.pixel.y, pt.pixel.z] = 0.0
                    fieldC2[pt.pixel.x, pt.pixel.y, pt.pixel.z] = 0.0

    def finish(self):
        """
        Finish Function is called after the last MCS
        """


        