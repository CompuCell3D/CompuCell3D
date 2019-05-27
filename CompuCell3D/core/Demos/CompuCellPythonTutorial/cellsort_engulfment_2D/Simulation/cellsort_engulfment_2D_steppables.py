from cc3d.core.PySteppables import *


class CellInitializer(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        size = 5
        x_min = 15
        x_max = 85
        y_min_low = 15
        y_max_low = 50

        y_min_high = 50
        y_max_high = 85

        for x in range(x_min, x_max, size):
            for y in range(y_min_low, y_max_low, size):
                self.cellField[x:x + size - 1, y:y + size - 1, 0] = self.new_cell(self.CONDENSING)

        for x in range(x_min, x_max, size):
            for y in range(y_min_high, y_max_high, size):
                self.cellField[x:x + size - 1, y:y + size - 1, 0] = self.new_cell(self.NONCONDENSING)
