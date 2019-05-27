from cc3d.core.PySteppables import *


class PixelTrackerSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        for cell in self.cell_list:
            if cell.type == 2:
                pixel_list = self.get_cell_pixel_list(cell)
                for pixel_tracker_data in pixel_list:
                    print("pixel of cell id=", cell.id, " type:", cell.type, " = ", pixel_tracker_data.pixel,
                          " number of pixels=", pixel_list.number_of_pixels())
