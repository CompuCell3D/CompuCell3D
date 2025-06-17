from cc3d.core.PySteppables import *
import numpy as np

class SelfReinforcingSteppable(SteppableBasePy):

    def __init__(self, frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        """
        Called before MCS=0 while building the initial simulation
        """
        
        cell_hwidth = 3
        self.cell_field[self.dim.x // 2 - cell_hwidth:self.dim.x // 2 + cell_hwidth, 
                        self.dim.y // 2 - cell_hwidth:self.dim.y // 2 + cell_hwidth, 0] = self.new_cell(self.cell_type.Cell)

    def step(self, mcs):
        """
        Called every frequency MCS while executing the simulation
        
        :param mcs: current Monte Carlo step
        """

        pass

    def finish(self):
        """
        Called after the last MCS to wrap up the simulation
        """

    def on_stop(self):
        """
        Called if the simulation is stopped before the last MCS
        """
