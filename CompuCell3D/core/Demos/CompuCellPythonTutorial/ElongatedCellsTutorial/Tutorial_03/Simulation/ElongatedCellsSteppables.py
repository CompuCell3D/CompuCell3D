from cc3d.core.PySteppables import *


class ElongatedCellsSteppable(SteppableBasePy):
    def __init__(self, frequency=1):

        SteppableBasePy.__init__(self, frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """
        top = self.new_cell(cell_type=1)
        self.cell_field[45:50, 25:30, 0] = top

        center_1 = self.new_cell(cell_type=2)
        self.cell_field[45:50, 30:35, 0] = center_1

        top_1 = self.new_cell(cell_type=1)
        self.cell_field[45:50, 35:40, 0] = top_1