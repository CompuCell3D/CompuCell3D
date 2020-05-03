
from cc3d.core.PySteppables import *

from math import sin

class FocalPointPlasticityOscillator2Steppable(SteppableBasePy):

    def __init__(self,frequency=10):

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
        link_life = 500
        dist_a = 10
        dist_o = 20
        sp = 4
        for cell in self.cell_list:
            for fppd in self.get_focal_point_plasticity_data_list(cell):
                n_cell = fppd.neighborAddress
                link_age = mcs - fppd.initMCS
                if link_age > link_life: # All links older than link_life are removed
                    self.focal_point_plasticity_plugin.deleteFocalPointPlasticityLink(cell, n_cell)
                else: # Link target distance oscillates over link_life
                    target_distance = dist_o + dist_a*sin(3.14*sp*link_age/link_life)
                    self.set_focal_point_plasticity_parameters(cell=cell,
                                                               n_cell=n_cell,
                                                               target_distance=target_distance)

    def finish(self):
        """
        Finish Function is called after the last MCS
        """
