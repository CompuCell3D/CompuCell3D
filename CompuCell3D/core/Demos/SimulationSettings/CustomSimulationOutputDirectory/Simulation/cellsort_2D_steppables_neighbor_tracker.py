from cc3d.core.PySteppables import *


class NeighborTrackerPrinterSteppable(SteppableBasePy):
    def __init__(self, frequency=100):

        SteppableBasePy.__init__(self, frequency)
        self.set_output_dir('neighbor_tracker_example_output_dir')

        # also a valid option
        # self.set_output_dir(r'c:\Users\m\CC3DWorkspace_1\neighbor_tracker_example_output_dir_2', abs_path=True)

    def step(self, mcs):
        
        for cell in self.cell_list:
            neighbor_list = self.get_cell_neighbor_data_list(cell)

            print("*********NEIGHBORS OF CELL WITH ID ", cell.id, " *****************")
            print("*********TOTAL NUMBER OF NEIGHBORS ", len(neighbor_list), " *****************")
            print("********* COMMON SURFACE AREA WITH TYPES (1,2) ",
                  neighbor_list.commonSurfaceAreaWithCellTypes(cell_type_list=[1, 2]), " *****************")
            print("********* COMMON SURFACE AREA BY TYPE ", neighbor_list.commonSurfaceAreaByType(),
                  " *****************")
            print("********* NEIGHBOR COUNT BY TYPE ", neighbor_list.neighborCountByType(), " *****************")

            for neighbor, common_surface_area in neighbor_list:
                if neighbor:
                    print("neighbor.id", neighbor.id, " common_surface_area=", common_surface_area)
                else:
                    print("Medium common_surface_area=", common_surface_area)
