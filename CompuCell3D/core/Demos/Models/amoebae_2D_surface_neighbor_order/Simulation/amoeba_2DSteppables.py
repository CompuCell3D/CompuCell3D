from cc3d.core.PySteppables import *


class amoeba_2DSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """

    def step(self, mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """
        for cell in self.cell_list:
            print(cell.id, cell.surface)

    def finish(self):
        """
        Finish Function is called after the last MCS
        """
