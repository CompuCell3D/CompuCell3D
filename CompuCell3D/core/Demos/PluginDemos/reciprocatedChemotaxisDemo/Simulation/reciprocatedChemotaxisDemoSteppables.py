from cc3d.core.PySteppables import *


class reciprocatedChemotaxisDemoSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """
        for cell in self.cell_list:
            cell.targetVolume = 49
            cell.lambdaVolume = 2

    def step(self, mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """
        if len(self.cell_list) == 0:
            self.stop_simulation()
        xfin = int(self.dim.x * 0.95)
        for cell in self.cell_list:
            if cell.xCOM > xfin:
                cell.targetVolume = 0

    def finish(self):
        """
        Finish Function is called after the last MCS
        """

    def on_stop(self):
        # this gets called each time user stops simulation
        return
