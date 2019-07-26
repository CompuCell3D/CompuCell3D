from cc3d.core.PySteppables import *


class VolumeParamSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):

        cell_0 = self.new_cell(self.CONDENSING)

        self.cell_field[0:6, 5:26, 0] = cell_0
        self.cell_field[90:100, 5:26, 0] = cell_0

        for cell in self.cell_list:
            cell.targetVolume = 25
            cell.lambdaVolume = 2.0

    def step(self, mcs):
        for cell in self.cell_list:
            cell.targetVolume += 1


class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)

    def step(self, mcs):

        cells_to_divide = []
        for cell in self.cell_list:
            if cell.volume > 50:
                cells_to_divide.append(cell)

        for cell in cells_to_divide:
            # to change mitosis mode leave one of the below lines uncommented
            self.divide_cell_random_orientation(cell)

            # Other valid options
            # self.divide_cell_orientation_vector_based(cell,1,1,0)
            # self.divide_cell_along_major_axis(cell)

    def update_attributes(self):
        # reducing parent target volume BEFORE clonning attributes
        self.parent_cell.targetVolume /= 2.0
        self.clone_parent_2_child()

        if self.parent_cell.type == self.CONDENSING:
            self.child_cell.type = self.NONCONDENSING
        else:
            self.child_cell.type = self.CONDENSING
