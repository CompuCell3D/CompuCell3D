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


class MitosisSteppableClusters(MitosisSteppableClustersBase):
    def __init__(self, frequency=1):
        MitosisSteppableClustersBase.__init__(self, frequency)

    def step(self, mcs):
        if mcs < 20:
            return

        mitosis_cluster_id_list = []
        for compartment_list in self.cluster_list:
            # print ("cluster has size=",compartment_list.size())
            cluster_id = 0
            cluster_volume = 0
            for cell in CompartmentList(compartment_list):
                cluster_volume += cell.volume
                cluster_id = cell.clusterId

            print("cluster_volume=", cluster_volume)

            # condition under which cluster mitosis takes place
            if cluster_volume > 250:
                # instead of doing mitosis right away we store ids for clusters which should be divide.
                # This avoids modifying cluster list while we iterate through it
                mitosis_cluster_id_list.append(cluster_id)

        for cluster_id in mitosis_cluster_id_list:

            self.divide_cluster_orientation_vector_based(cluster_id, 1, 0, 0)

            # valid options - to change mitosis mode leave one of the below lines uncommented
            # self.divide_cluster_random_orientation(cluster_id)
            # self.divide_cluster_along_major_axis(cluster_id)
            # self.divide_cluster_along_minor_axis(cluster_id)

    def update_attributes(self):
        # compartments in the parent and child clusters are
        # listed in the same order so attribute changes require simple iteration through compartment list
        compartment_list_parent = self.get_cluster_cells(self.parent_cell.clusterId)

        for i in range(len(compartment_list_parent)):
            compartment_list_parent[i].targetVolume /= 2.0
        self.clone_parent_cluster_2_child_cluster()
