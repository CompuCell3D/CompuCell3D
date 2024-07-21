from cc3d.core.PySteppables import *


class CellAttributeTracker(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.plot_win = None

    def start(self):
        #Create the histogram
        self.plot_win = self.add_new_plot_window(title='Histogram of Cell Volumes', x_axis_title='Volume Size in Pixels',
                                                y_axis_title='Number of Cells')
        self.plot_win.add_histogram_plot(plot_name='VolumeHistogram1', color='green', alpha=100)

    def step(self, mcs):
        #Update histogram
        hist_list = [1] * len(self.cell_list)
        for i, cell in enumerate(self.cell_list):
            hist_list[i] = cell.volume
        self.plot_win.add_histogram(plot_name='VolumeHistogram1', value_array=hist_list, number_of_bins=10)