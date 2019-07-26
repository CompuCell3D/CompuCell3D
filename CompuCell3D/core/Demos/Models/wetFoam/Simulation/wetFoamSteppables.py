from cc3d.core.PySteppables import *
from random import randint


class FlexCellInitializer(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.cell_type_parameters = {}
        self.water_fraction = 0.1
        self.medium_cell = None

    def add_cell_type_parameters(self, cell_type, count, target_volume, lambda_volume):
        self.cell_type_parameters[cell_type] = [count, target_volume, lambda_volume]

    def set_fraction_of_water(self, water_fraction):
        if 0.0 < water_fraction < 1.0:
            self.water_fraction = water_fraction

    def start(self):
        self.add_cell_type_parameters(cell_type=1, count=0, target_volume=25, lambda_volume=10.0)
        self.add_cell_type_parameters(cell_type=2, count=0, target_volume=5, lambda_volume=2.0)

        number_of_pixels = self.dim.x * self.dim.y

        count_type1 = int(number_of_pixels * (1 - self.water_fraction) / self.cell_type_parameters[1][1])
        count_type2 = int(number_of_pixels * self.water_fraction / self.cell_type_parameters[2][1])

        self.cell_type_parameters[1][0] = count_type1
        self.cell_type_parameters[2][0] = count_type2

        self.medium_cell = self.cell_field[0, 0, 0]

        for cell_type, cell_type_param_list in self.cell_type_parameters.items():
            # initialize self.cellTypeParameters[0]+1 number of randomly placed cells with user
            # specified targetVolume and lambdaVolume
            for cell_count in range(cell_type_param_list[0]):
                cell = self.potts.createCell()
                self.cellField[
                    randint(0, self.dim.x - 1), randint(0, self.dim.y - 1), randint(0, self.dim.z - 1)] = cell

                cell.type = cell_type
                cell.targetVolume = cell_type_param_list[1]
                cell.lambdaVolume = cell_type_param_list[2]

    def step(self, mcs):
        if mcs == 300:
            for cell in self.cell_list:
                if cell.type == 1:
                    cell.lambdaVolume = 0.0
                if cell.type == 2:
                    cell.lambdaVolume = 10.0

        if mcs == 200:

            # fill medium with water
            for x, y, z in self.every_pixel():
                current_cell = self.cellField[x, y, z]

                if not current_cell:
                    cell = self.potts.createCell()
                    self.cellField[x, y, z] = cell
                    cell.type = 2
                    cell.targetVolume = self.cell_type_parameters[2][1]
                    cell.lambdaVolume = self.cell_type_parameters[2][2]
