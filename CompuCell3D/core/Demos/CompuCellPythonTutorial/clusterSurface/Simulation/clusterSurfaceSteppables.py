from cc3d.core.PySteppables import *


class VolumeParamSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        for cell in self.cell_list:
            cell.targetVolume = 25
            cell.lambdaVolume = 2.0

    def step(self, mcs):
        for cell in self.cell_list:
            cell.targetVolume += 1

        for compartments in self.clusters:
            for cell in compartments:
                self.clusterSurfacePlugin.setTargetAndLambdaClusterSurface(cell, 80, 2.0)
                break


class MitosisSteppableClusters(MitosisSteppableClustersBase):
    def __init__(self, frequency=1):
        MitosisSteppableClustersBase.__init__(self, frequency)

    def step(self, mcs):

        for cell in self.cell_list:
            cluster_cell_list = self.get_cluster_cells(cell.clusterId)
            print("DISPLAYING CELL IDS OF CLUSTER ", cell.clusterId, "CELL. ID=", cell.id)
            for cell_local in cluster_cell_list:
                print("CLUSTER CELL ID=", cell_local.id, " type=", cell_local.type)
                print('cluster_surface=', cell_local.clusterSurface)

        for compartments in self.clusters:
            cluster_id = -1
            cluster_cell = None
            cluster_surface = 0.0
            for cell in compartments:
                cluster_cell = cell
                cluster_id = cell.clusterId
                for pixel_tracker_data in self.get_cell_pixel_list(cell):
                    for neighbor in self.get_pixel_neighbors_based_on_neighbor_order(
                            pixel=pixel_tracker_data.pixel, neighbor_order=1):

                        n_cell = self.cellField.get(neighbor.pt)
                        if not n_cell:
                            # only medium contributes in this case
                            cluster_surface += 1.0
                        elif cell.clusterId != n_cell.clusterId:
                            cluster_surface += 1.0

            print('MANUAL CALCULATION cluster_id=', cluster_id, ' cluster_surface=', cluster_surface)
            print('AUTOMATIC UPDATE cluster_id=', cluster_id, ' cluster_surface=', cluster_cell.clusterSurface)

        if mcs == 400:
            cell1 = None
            for cell in self.cell_list:
                cell1 = cell
                break
            self.reassign_cluster_id(cell1, 2)

        mitosis_cluster_id_list = []
        for compartmentList in self.cluster_list:

            cluster_id = 0
            cluster_volume = 0
            for cell in CompartmentList(compartmentList):
                cluster_volume += cell.volume
                cluster_id = cell.clusterId

            # condition under which cluster mitosis takes place
            if cluster_volume > 250:
                # instead of doing mitosis right away we store ids for clusters which should be divide.
                # This avoids modifying cluster list while we iterate through it
                mitosis_cluster_id_list.append(cluster_id)

        for cluster_id in mitosis_cluster_id_list:
            self.divide_cluster_random_orientation(cluster_id)

            # valid options - to change mitosis mode leave one of the below lines uncommented
            # self.divide_cluster_orientation_vector_based(cluster_id, 1, 0, 0)
            # self.divide_cluster_along_major_axis(cluster_id)
            # self.divide_cluster_along_minor_axis(cluster_id)

    def update_attributes(self):
        # compartments in the parent and child clusters arel listed in the same order
        # so attribute changes require simple iteration through compartment list
        parent_cell = self.mitosisSteppable.parentCell
        child_cell = self.mitosisSteppable.childCell

        compartment_list_child = self.inventory.getClusterCells(child_cell.clusterId)
        compartment_list_parent = self.inventory.getClusterCells(parent_cell.clusterId)
        print("compartment_list_child=", compartment_list_child)
        for i in range(len(compartment_list_child)):
            compartment_list_parent[i].targetVolume /= 2.0
            # compartment_list_parent[i].targetVolume=25
            compartment_list_child[i].targetVolume = compartment_list_parent[i].targetVolume
            compartment_list_child[i].lambdaVolume = compartment_list_parent[i].lambdaVolume
