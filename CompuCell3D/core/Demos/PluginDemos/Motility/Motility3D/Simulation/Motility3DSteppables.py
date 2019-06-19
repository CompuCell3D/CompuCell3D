
from cc3d.core.PySteppables import *

import numpy as np
import numpy.linalg as nalg

class Motility3DSteppable(SteppableBasePy):

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

        for cell in self.cell_list:

            print("\n cell.biasVecX = ",cell.biasVecX)
            print("\n cell.biasVecY = ",cell.biasVecY)
            print("\n cell.biasVecZ = ",cell.biasVecZ)
            print("\n norm = ", nalg.norm([cell.biasVecX,cell.biasVecY,cell.biasVecZ]))
            break

    def finish(self):
        """
        Finish Function is called after the last MCS
        """


        