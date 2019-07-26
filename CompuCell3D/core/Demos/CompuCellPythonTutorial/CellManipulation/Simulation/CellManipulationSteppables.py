from cc3d.core.PySteppables import *

class CellManipulationSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        if mcs == 10:

            # we can use CompuCell.Point3D to specify shift vector
            # shift_vector = CompuCell.Point3D(20, 20, 0)
            shift_vector = [20, 20, 0]
            for cell in self.cell_list:
                self.move_cell(cell, shift_vector)

        if mcs == 20:
            self.cell_field[50:55,50:55,0] = self.new_cell(self.NONCONDENSING)

        if mcs == 30:
            for cell in self.cell_list:
                self.delete_cell(cell)
