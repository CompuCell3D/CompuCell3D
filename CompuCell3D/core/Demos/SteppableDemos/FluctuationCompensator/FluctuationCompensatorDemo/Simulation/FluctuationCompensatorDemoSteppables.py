from cc3d.core.PySteppables import *
from cc3d.cpp import CompuCell
import random


class FluctuationCompensatorDemoSteppable(SteppableBasePy):

    def __init__(self, frequency=1):

        SteppableBasePy.__init__(self, frequency)

        self.yes_fluc_comp = None
        self.no_fluc_comp = None

    def start(self):
        self.yes_fluc_comp = CompuCell.getConcentrationField(self.simulator, "YesFlucComp")
        self.no_fluc_comp = CompuCell.getConcentrationField(self.simulator, "NoFlucComp")

    def step(self, mcs):
        if self.pixel_tracker_plugin is None:
            return

        # Test modification of field values outside of core routines
        mcs_change_rate = 10e3
        if mcs % mcs_change_rate == 0:
            for cell in self.cell_list:
                new_val = random.random()
                for ptd in self.get_cell_pixel_list(cell):
                    self.yes_fluc_comp[ptd.pixel.x, ptd.pixel.y, 0] = new_val
                    self.no_fluc_comp[ptd.pixel.x, ptd.pixel.y, 0] = new_val

            # Without this call, modifications (other than by core routines) to a field with a solver using
            # FluctuationCompensator will likely cause numerical errors
            CompuCell.updateFluctuationCompensators()

    def finish(self):
        pass

