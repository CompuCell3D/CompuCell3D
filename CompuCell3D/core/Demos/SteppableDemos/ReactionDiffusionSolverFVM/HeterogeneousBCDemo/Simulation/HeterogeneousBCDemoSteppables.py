
from cc3d.core.PySteppables import *

from cc3d.cpp import CompuCell


class HeterogeneousBCDemoSteppable(SteppableBasePy):

    def __init__(self, frequency=1):

        SteppableBasePy.__init__(self, frequency)

    def start(self):
        """
        Called before MCS=0 while building the initial simulation
        """

        # Make the top-right of Field 1 and bottom-left of Field 2 = 1,
        # with linear functions to zero on neighboring boundaries
        for x in range(self.dim.x):
            cf = x / (self.dim.x - 1)
            self.reaction_diffusion_solver_fvm.useFixedConcentration("Field1", "MaxY",
                                                                     cf, CompuCell.Point3D(x, self.dim.y - 1, 0))
            self.reaction_diffusion_solver_fvm.useFixedConcentration("Field2", "MinY",
                                                                     1.0 - cf, CompuCell.Point3D(x, 0, 0))
        for y in range(self.dim.y):
            cf = y / (self.dim.y - 1)
            self.reaction_diffusion_solver_fvm.useFixedConcentration("Field1", "MaxX",
                                                                     cf, CompuCell.Point3D(self.dim.x - 1, y, 0))
            self.reaction_diffusion_solver_fvm.useFixedConcentration("Field2", "MinX",
                                                                     1.0 - cf, CompuCell.Point3D(0, y, 0))

    def step(self, mcs):
        """
        Called every frequency MCS while executing the simulation
        
        :param mcs: current Monte Carlo step
        """

    def finish(self):
        """
        Called after the last MCS to wrap up the simulation
        """

    def on_stop(self):
        """
        Called if the simulation is stopped before the last MCS
        """
