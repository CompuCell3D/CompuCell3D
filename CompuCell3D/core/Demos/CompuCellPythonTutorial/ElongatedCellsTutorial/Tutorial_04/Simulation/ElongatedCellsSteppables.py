from cc3d.core.PySteppables import *


class ElongatedCellsSteppable(SteppableBasePy):
    def __init__(self, frequency=1):

        SteppableBasePy.__init__(self, frequency)

    def start(self):
        self.create_arranged_cells(x_s=25, y_s=25, size=5, cell_type_ids=[1, 2, 2, 2, 2, 1])
        for cell in self.cell_list:
            print("cell id=", cell.id, " clusterId=", cell.clusterId)

    def create_arranged_cells(self, x_s, y_s, size, cell_type_ids=None):
        """
        this function creates vertically arranged cells.

        x_s, ys - coordinates of bottom_left corner of the cell arrangement
        size - size of the cell arrangement
        cell_type_ids - list of cell type ids

        """
        for i, cell_type_id in enumerate(cell_type_ids):
            cell = self.new_cell(cell_type=cell_type_id)
            self.cell_field[x_s : x_s + size, y_s + i * size : y_s + (i + 1) * size, 0] = cell
