from cc3d.core.PySteppables import *


class VolumeParamSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):

        self.cell_field[5:11,5:11,0] = self.new_cell(self.CONDENSING)

        for cell in self.cell_list:
            cell.targetVolume = 25
            cell.lambdaVolume = 2.0

    def step(self, mcs):
        for cell in self.cellList:
            cell.targetVolume += 1


class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)

        # 0 - parent child position will be randomized between mitosis event
        # negative integer - parent appears on the 'left' of the child
        # positive integer - parent appears on the 'right' of the child
        self.set_parent_child_position_flag(-1)

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
            # self.divide_cell_along_minor_axis(cell)

    def update_attributes(self):

        # reducing parent target volume BEFORE cloning
        self.parent_cell.targetVolume /= 2.0

        self.clone_parent_2_child()

        # implementing
        if self.parent_cell.type == self.CONDENSING:
            self.child_cell.type = self.NONCONDENSING
        else:
            self.child_cell.type = self.CONDENSING
