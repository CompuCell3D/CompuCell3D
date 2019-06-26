from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
import numpy as np
from scipy import special, optimize
from scipy import integrate
from numpy import exp
from scipy import stats


class InventoryCheckSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency=frequency)

    def start(self):
        print("INSIDE START FUNCTION")
        print('view_manager = ', CompuCellSetup.persistent_globals.view_manager)

        self.plot_win = self.add_new_plot_window(title='Average Volume And Surface',
                                                 x_axis_title='MonteCarlo Step (MCS)',
                                                 y_axis_title='Variables', x_scale_type='linear', y_scale_type='linear',
                                                 grid=False)

        self.plot_win.add_plot("MVol", style='Lines', color='red')
        self.plot_win.add_plot("MSur", style='Dots', color='green')

    def step(self, mcs):

        if not self.plot_win:
            print("To get scientific plots working you need extra packages installed: numpy pyqtgraph")
            return

        # self.pW.addDataPoint("MCS1",mcs,-2*mcs)
        # this is non optimized code. It is for illustrative purposes only.

        f = lambda x: exp(-x ** 2)
        i = integrate.quad(f, 0, 1)

        print("i=", i)

        mean_surface = 0.0
        mean_volume = 0.0
        number_of_cells = 0
        for cell in self.cell_list:
            mean_volume += cell.volume
            mean_surface += cell.surface
            number_of_cells += 1
        mean_volume /= float(number_of_cells)
        mean_surface /= float(number_of_cells)

        if 100 < mcs < 200:
            self.plot_win.erase_all_data()
        else:
            self.plot_win.add_data_point("MVol", mcs, mean_volume)
            self.plot_win.add_data_point("MSur", mcs, mean_surface)
            if mcs >= 200:
                print("Adding meanVolume=", mean_volume)
                try:
                    print("plotData=", self.plot_win.plot_data["MVol"])
                except TypeError:
                    # happens in CML mode
                    pass

        self.plot_win.show_all_plots()

        if mcs == 300:
            self.stop_simulation()
