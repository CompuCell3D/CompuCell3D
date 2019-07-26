from cc3d.core.PySteppables import *
from pathlib import Path


class ExtraPlotSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.plot_win = None

    def start(self):

        config_options = {'background': 'white', 'legend': True}

        self.plot_win = self.add_new_plot_window(title='Average Volume And Surface',
                                                 x_axis_title='MonteCarlo Step (MCS)',
                                                 y_axis_title='Variables', x_scale_type='linear', y_scale_type='linear',
                                                 config_options=config_options)

        self.plot_win.add_plot('MVol', style='Dots', color='red', _size=5)
        self.plot_win.add_plot('MSur', style='Bars', size=0.2)

    def step(self, mcs):

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
            self.plot_win.add_data_point('MVol', mcs, mean_volume)
            self.plot_win.add_data_point('MSur', mcs, mean_surface)
            if mcs >= 200:
                print('Adding meanVolume=', mean_volume)
                print('plotData=', self.plot_win.plotData['MVol'])

        # Saving plots as PNG's
        if mcs < 50 and self.output_dir is not None:
            png_out_path = Path(self.output_dir).joinpath('ExtraPlots_' + str(mcs) + '.png')

            # here we specify size of the image saved - default is 400 x 400
            self.plot_win.save_plot_as_png(str(png_out_path), 1000, 1000)

            out_path = Path(self.output_dir).joinpath('ExtraPlots_' + str(mcs) + '.txt')
            self.plot_win.save_plot_as_data(str(out_path), CSV_FORMAT)


class ExtraMultiPlotSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

        self.plot_win_vol = None
        self.plot_win_sur = None

    def start(self):
        config_options_1 = {'background': 'white111', 'legend': False}
        config_options_2 = {'background': 'green', 'legend': True}

        self.plot_win_vol = self.add_new_plot_window(title='Average Volume', x_axis_title='MonteCarlo Step (MCS)',
                                                     y_axis_title='Average Volume', config_options=config_options_1)
        self.plot_win_vol.add_plot(plot_name='MVol', style='Dots', color='red', size=5)

        self.plot_win_sur = self.add_new_plot_window(title='Average Surface', x_axis_title='MonteCarlo Step (MCS)',
                                                     y_axis_title='Average Surface', config_options=config_options_2)
        self.plot_win_sur.add_plot(plot_name='MSur')

    def step(self, mcs):
        mean_surface = 0.0
        mean_volume = 0.0
        number_of_cells = 0
        for cell in self.cellList:
            mean_volume += cell.volume
            mean_surface += cell.surface
            number_of_cells += 1
        mean_volume /= float(number_of_cells)
        mean_surface /= float(number_of_cells)

        self.plot_win_vol.add_data_point("MVol", mcs, mean_volume)
        self.plot_win_sur.add_data_point("MSur", mcs, mean_surface)
        print("meanVolume=", mean_volume, "meanSurface=", mean_surface)
