from cc3d.core.PySteppables import *


class TargetVolumeDrosoSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):

        for cell in self.cell_list:
            cell.targetVolume = 25.0
            cell.lambdaVolume = 2.0

    def step(self, mcs):
        for cell in self.cell_list:
            if ((cell.xCOM - 100) ** 2 + (cell.yCOM - 100) ** 2) < 400:
                cell.targetVolume += 1


class CellKiller(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        for cell in self.cell_list:
            if mcs == 10:
                cell.targetVolume = 0


class PressureFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.create_scalar_field_cell_level_py("PressureField")

    def step(self, mcs):
        pressure_field = self.field.PressureField
        pressure_field.clear()
        for cell in self.cell_list:
            pressure_field[cell] = cell.targetVolume - cell.volume
