from cc3d.core.PySteppables import *


class UniformInitializer(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        print("UniformInitializer: This function is called once before simulation")
        size = 5
        x_min = 10
        x_max = 80
        y_min = 10
        y_max = 80

        for x in range(x_min, x_max, size):
            for y in range(y_min, y_max, size):
                self.cell_field[x:x + size - 1, y:y + size - 1, 0] = self.new_cell(self.CONDENSING)

        cell0 = self.cell_field[0, 0, 0]
        cell1 = self.cell_field[50, 50, 0]
        print('cells=', (cell0, cell1))
        self.cell_field[50:80, 50:80, 0] = cell0
