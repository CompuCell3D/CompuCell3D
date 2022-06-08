from cc3d.core.PySteppables import *
from pathlib import Path


class ExtraPlotSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.clear_flag = False
        self.plot_win = None

    def start(self):

        self.plot_win = self.add_new_plot_window(title='Average Volume And Surface',
                                                 x_axis_title='MonteCarlo Step (MCS)',
                                                 y_axis_title='Variables', x_scale_type='linear', y_scale_type='linear',
                                                 grid=False, config_options={'legend': True})

        self.plot_win.add_plot("MVol", style='Lines', color='red', separate_y_axis=True,
                               # y_min=0.0,
                               y_max=100.0,
                               # y_scale_type='log'
                               )
        self.plot_win.add_plot("MSur_2", style='Dots', color='blue', separate_y_axis=True, y_min=0, y_max=5)
        #
        self.plot_win.add_plot("MSur", style='Dots', color='green', separate_y_axis=True, y_max=20.0)
        self.plot_win.add_plot("MVol_2", style='Lines', color='orange', separate_y_axis=True,
                               y_scale_type='log'
                               )
        self.plot_win.add_plot("MVol_3", style='Lines', color='yellow', separate_y_axis=True, y_max=0.5)

        print()

    def step(self, mcs):
        if not self.plot_win:
            return

        mean_surface = 0.0
        mean_volume = 0.0
        number_of_cells = 0
        for cell in self.cellList:
            mean_volume += cell.volume
            mean_surface += cell.surface
            number_of_cells += 1
        mean_volume /= float(number_of_cells)
        mean_surface /= float(number_of_cells)

        if mcs > 100 and mcs < 200:
            self.plot_win.erase_all_data()
        else:
            print("MVol_2 ", mcs, np.sin(mean_volume / 10.0))
            print("MVol_3 ", mcs, np.exp(mean_volume / 20.0))
            self.plot_win.add_data_point("MVol", mcs, mean_volume / 5.0)
            self.plot_win.add_data_point("MVol_2", mcs, np.sin(mean_volume / 10.0))
            # self.plot_win.add_data_point("MVol_3", mcs, np.exp(mean_volume / 20.0))
            self.plot_win.add_data_point("MVol_3", mcs, np.sin(mcs))

            self.plot_win.add_data_point("MSur", mcs, mean_surface)
            self.plot_win.add_data_point("MSur_2", mcs, mean_surface + mean_volume)
            if mcs >= 200:
                print("Adding meanVolume=", mean_volume)
                print("plotData=", self.plot_win.plotData["MVol"])

        # Saving plots as PNG's
        if mcs < 50 and self.output_dir is not None:
            file_name = str(Path(self.output_dir).joinpath("ExtraPlots_" + str(mcs) + ".png"))
            # here we specify size of the image saved - default is 400 x 400
            self.plot_win.save_plot_as_png(file_name, 550, 550)


