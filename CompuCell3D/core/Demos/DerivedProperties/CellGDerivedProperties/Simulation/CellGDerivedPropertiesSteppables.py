from cc3d.core.PySteppables import *
from cc3d.cpp.CompuCell import CellG


class CellGDerivedPropertiesSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """
        for cell in self.cell_list:
            cell.lambdaVolume, cell.targetVolume = 2.0, 50.0
            cell.lambdaSurface, cell.targetSurface = 2.0, 50.0

            self.reassign_cluster_id(cell, int(cell.id / 2 - 0.1))

        for cluster in self.clusters:
            for cell in cluster:
                cell.lambdaClusterSurface, cell.targetClusterSurface = 2.0, 50.0

    def step(self, mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """
        for cell in self.cell_list:
            print(f"cell id: {cell.id}")
            # Derived property: internal pressure
            print("cell volume, target volume, volume lambda:",
                  cell.volume, cell.targetVolume, cell.lambdaVolume)
            print(f"   internal pressure: {cell.pressure}")
            # Derived property: surface tension
            print("cell surface, target surface, surface lambda:",
                  cell.surface, cell.targetSurface, cell.lambdaSurface)
            print(f"   surface tension: {cell.surfaceTension}")

        for cluster in self.clusters:
            for cell in cluster:
                print(f"cluster id: {cell.clusterId}")
                # Derived property: cluster surface tension
                print("cluster surface area, target surface area, surface lambda:",
                      cell.clusterSurface, cell.targetClusterSurface, cell.lambdaClusterSurface)
                print(f"   cluster surface tension: {cell.clusterSurfaceTension}")

    def finish(self):
        """
        Finish Function is called after the last MCS
        """

    def on_stop(self):
        # this gets called each time user stops simulation
        return
