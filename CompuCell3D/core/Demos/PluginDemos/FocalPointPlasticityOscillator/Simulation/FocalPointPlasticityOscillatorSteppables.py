
from cc3d.core.PySteppables import *

class FocalPointPlasticityOscillatorSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)
        

    def start(self):
        """
        any code in the start function runs before MCS=0
        """
        self.id_oi = 1
        self.param_target_distances = [5, 30]
        self.cells_no_links = [cell for cell in self.cell_list if cell.id != self.id_oi]
        self.cell_oi = [cell for cell in self.cell_list if cell.id == self.id_oi][0]

    def step(self,mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """
        
        # Manually remove all links except for cell with id of interest
        for cell in self.cells_no_links:
            for fppd in self.get_focal_point_plasticity_data_list(cell):
                n_cell = fppd.neighborAddress
                if n_cell.id != self.id_oi:
                    self.focal_point_plasticity_plugin.deleteFocalPointPlasticityLink(cell, n_cell)

        # Count all links for cell of interest
        print("cell.id=",self.cell_oi.id)
        print("   Number of links: " + str(self.get_focal_point_plasticity_num_neighbors(self.cell_oi)))
        for fppd in self.get_focal_point_plasticity_data_list(self.cell_oi):
            print("   Target distance: " + str(fppd.targetDistance))
            print("   Is initiator : " + str(fppd.isInitiator))
            print("   Initiator id: " + str(self.get_focal_point_plasticity_initiator(self.cell_oi, fppd.neighborAddress).id))

        # Alternate target distance
        if mcs % 100 == 0:
            param_target_distance = self.param_target_distances[0]
            self.param_target_distances.reverse()
            self.set_focal_point_plasticity_parameters(self.cell_oi, target_distance=param_target_distance)

    def finish(self):
        """
        Finish Function is called after the last MCS
        """
