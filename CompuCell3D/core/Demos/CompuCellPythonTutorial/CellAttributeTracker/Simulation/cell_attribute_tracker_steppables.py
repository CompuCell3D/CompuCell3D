from cc3d.core.PySteppables import *


class CellAttributeTracker(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        self.track_cell_level_scalar_attribute("cell_volume", "volume")
        self.track_cell_level_scalar_attribute("cell_volume_sq", "volume", function_obj=lambda x: x ** 2)
        self.track_cell_level_vector_attribute("my_vector_plot", "my_vector")
        self.track_cell_level_vector_attribute("my_vector_plot_function", "my_vector",
                                               function_obj=lambda vec: [vec[0] ** 2, vec[1] / 2, vec[2] * 2])

        self.histogram_scalar_attribute(histogram_name="Volume", attribute_name="volume", number_of_bins=20,
                                        function=lambda x: x ** 2, cell_type_list=[self.CONDENSING],
                                        x_axis_title='vol**2',
                                        y_axis_title='Count', color='green', x_scale_type='linear',
                                        y_scale_type='linear')

    def step(self, mcs):
        for cell in self.cell_list:
            cell.dict['my_vector'] = [cell.volume, cell.volume, cell.volume / 2.0]
