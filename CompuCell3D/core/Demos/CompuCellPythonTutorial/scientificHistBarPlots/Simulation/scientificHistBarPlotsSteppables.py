from cc3d.core.PySteppables import *
import random
import numpy as np
from pathlib import Path


class HistPlotSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.plot_win = None

    def start(self):

        # initialize setting for Histogram
        self.plot_win = self.add_new_plot_window(title='Histogram of Cell Volumes', x_axis_title='Number of Cells',
                                                 y_axis_title='Volume Size in Pixels')
        # alpha is transparency 0 is transparent, 255 is opaque
        self.plot_win.add_histogram_plot(plot_name='Hist 1', color='green', alpha=100)
        self.plot_win.add_histogram_plot(plot_name='Hist 2', color='red', alpha=100)
        self.plot_win.add_histogram_plot(plot_name='Hist 3', color='blue')

    def step(self, mcs):

        vol_list = []
        for cell in self.cell_list:
            vol_list.append(cell.volume)

        gauss = np.random.normal(0.0, 1.0, size=(100,))

        self.plot_win.add_histogram(plot_name='Hist 1', value_array=gauss, number_of_bins=10)
        self.plot_win.add_histogram(plot_name='Hist 2', value_array=vol_list, number_of_bins=10)
        self.plot_win.add_histogram(plot_name='Hist 3', value_array=vol_list, number_of_bins=50)

        if self.output_dir is not None:
            output_path = Path(self.output_dir).joinpath("HistPlots_" + str(mcs) + ".txt")
            self.plot_win.save_plot_as_data(output_path, CSV_FORMAT)

            png_output_path = Path(self.output_dir).joinpath("HistPlots_" + str(mcs) + ".png")

            # here we specify size of the image saved - default is 400 x 400
            self.plot_win.save_plot_as_png(png_output_path, 1000, 1000)


class BarPlotSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.plot_win = None

    def start(self):

        self.plot_win = self.add_new_plot_window(title='Bar Plot', x_axis_title='Growth of US GDP',
                                                 y_axis_title='Number of Suits')
        self.plot_win.add_plot(plot_name='GDP', color='red', style='bars', size=0.5)

    def step(self, mcs):

        if mcs % 20 == 0:

            self.plot_win.erase_all_data()

            gdp_list = random.sample(range(1, 100), 6)
            locations = random.sample(range(1, 20), 6)

            self.plot_win.add_data_series('GDP', locations, gdp_list)

        if self.output_dir is not None:
            png_output_path = Path(self.output_dir).joinpath("BarPlots_" + str(mcs) + ".png")

            # here we specify size of the image saved - default is 400 x 400
            self.plot_win.save_plot_as_png(png_output_path, 1000, 1000)

            output_path = Path(self.output_dir).joinpath("BarPlots_" + str(mcs) + ".txt")
            self.plot_win.save_plot_as_data(output_path, CSV_FORMAT)
