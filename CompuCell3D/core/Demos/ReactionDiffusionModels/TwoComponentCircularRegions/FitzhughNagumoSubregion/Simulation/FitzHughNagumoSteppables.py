from cc3d.cpp.PlayerPython import * 
from cc3d import CompuCellSetup

from cc3d.core.PySteppables import *

class FitzHughNagumoSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """
        pass

    def step(self,mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """
        pass
        
        
        
        

    def finish(self):
        """
        Finish Function is called after the last MCS
        """


