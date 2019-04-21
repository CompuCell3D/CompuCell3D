from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
import numpy as np
from scipy import special, optimize

class InventoryCheckSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency=frequency)

    def start(self):
        print("INSIDE START FUNCTION")
        print('view_manager = ', CompuCellSetup.persistent_globals.view_manager)

        self.pW = self.add_new_plot_window(title='Average Volume And Surface', x_axis_title='MonteCarlo Step (MCS)',
                                           y_axis_title='Variables', x_scale_type='linear', y_scale_type='linear', grid=False)

        self.pW.addPlot("MVol", _style='Lines', _color='red')
        self.pW.addPlot("MSur", _style='Dots', _color='green')

        # adding automatically generated legend
        # default position is at the bottom of the plot but here we put it at the top
        self.pW.addAutoLegend("top")

    def step(self, mcs):

        if not self.pW:
            print("To get scientific plots working you need extra packages installed: numpy pyqtgraph")
            return

        # self.pW.addDataPoint("MCS1",mcs,-2*mcs)
        # this is non optimized code. It is for illustrative purposes only.

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
            self.pW.eraseAllData()
        else:
            self.pW.addDataPoint("MVol", mcs, mean_volume)
            self.pW.addDataPoint("MSur", mcs, mean_surface)
            if mcs >= 200:
                print("Adding meanVolume=", mean_volume)
                print("plotData=", self.pW.plotData["MVol"])

        self.pW.showAllPlots()

        if mcs == 300:
            self.stop_simulation()
