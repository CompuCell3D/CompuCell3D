from cc3d.core.PySteppables import *


class BoundaryPixelTrackerSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.create_scalar_field_py("BOUNDARY")

    def step(self, mcs):
        for cell in self.cellList:

            pixel_list = self.get_cell_boundary_pixel_list(cell, 3)
            boundary_field = self.field.BOUNDARY
            if cell.id == 8:
                for boundary_pixel_tracker_data in pixel_list:
                    print("pixel of cell id=", cell.id, " type:", cell.type, " = ", boundary_pixel_tracker_data.pixel,
                          " number of pixels=", pixel_list.number_of_pixels())
                    pt = boundary_pixel_tracker_data.pixel
                    boundary_field[pt.x, pt.y, pt.z] = 20
