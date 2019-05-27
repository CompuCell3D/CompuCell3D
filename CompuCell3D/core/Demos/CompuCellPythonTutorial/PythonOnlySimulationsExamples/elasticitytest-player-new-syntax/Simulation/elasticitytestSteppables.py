from cc3d.core.PySteppables import *


class ElasticityLocalSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

        self.links_initialized = False

    def initialize_elasticity_local(self):

        for cell in self.cellList:

            elasticity_data_list = self.get_elasticity_data_list(cell)
            for elasticity_data in elasticity_data_list:  # visiting all elastic links of 'cell'

                target_length = elasticity_data.targetLength
                elasticity_data.targetLength = 6.0
                elasticity_data.lambdaLength = 200.0
                elasticity_neighbor = elasticity_data.neighborAddress

                # now we set up elastic link data stored in neighboring cell
                neighbor_elasticity_data = None
                neighbor_elasticity_data_list = self.get_elasticity_data_list(elasticity_neighbor)
                for neighbor_elasticity_data_tmp in neighbor_elasticity_data_list:
                    if not CompuCell.areCellsDifferent(neighbor_elasticity_data_tmp.neighborAddress, cell):
                        neighbor_elasticity_data = neighbor_elasticity_data_tmp
                        break

                if neighbor_elasticity_data is None:
                    raise RuntimeError("None Type returned. Problems with FemDataNeighbors initialization or sets of "
                           "neighbor_elasticity_data are corrupted")
                neighbor_elasticity_data.targetLength = 6.0
                neighbor_elasticity_data.lambdaLength = 200.0

    def step(self, mcs):
        if not self.links_initialized:
            self.initialize_elasticity_local()
            # adding link between cell.id=1 and cell.id=3
            cell1 = None
            cell3 = None
            for cell in self.cellList:
                if cell.id == 1:
                    cell1 = cell
                if cell.id == 3:
                    cell3 = cell
            self.elasticity_tracker_plugin.addNewElasticLink(cell1, cell3, 200.0, 6.0)
