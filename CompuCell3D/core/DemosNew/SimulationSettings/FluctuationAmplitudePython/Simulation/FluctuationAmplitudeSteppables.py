from cc3d.core.PySteppables import *


class FluctuationAmplitude(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.quarters = [[0, 0, 50, 50], [0, 50, 50, 100], [50, 50, 100, 100], [50, 0, 100, 50]]
        self.steppable_call_counter = 0

    def step(self, mcs):

        quarter_index = self.steppable_call_counter % 4

        quarter = self.quarters[quarter_index]
        for cell in self.cell_list:

            if quarter[0] <= cell.xCOM < quarter[2] and quarter[1] <= cell.yCOM < quarter[3]:
                cell.fluctAmpl = 50
            else:
                # this means CompuCell3D will use globally defined FluctuationAmplitude
                cell.fluctAmpl = -1

        self.steppable_call_counter += 1
