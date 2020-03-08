from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
from random import random

class ScreenshotSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):

        if mcs in [3, 5, 19,20, 23, 29, 31]:
            self.request_screenshot(mcs=mcs, screenshot_label='Cell_Field_CellField_2D_XY_0')

            
        