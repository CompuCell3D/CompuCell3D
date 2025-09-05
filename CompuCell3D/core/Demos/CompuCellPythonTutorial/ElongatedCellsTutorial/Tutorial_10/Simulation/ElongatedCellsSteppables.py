from cc3d.core.PySteppables import *
import random


class ElongatedCellsSteppable(SteppableBasePy):
    def __init__(self, frequency=1):

        SteppableBasePy.__init__(self, frequency)
        self.leading_cells_ids = set()
        self.maxAbsLambdaX = 10

    def start(self):
        self.create_arranged_cells(x_s=25, y_s=25, size=5, cell_type_ids=[1, 2, 2, 2, 2, 1])
        self.create_arranged_cells(x_s=40, y_s=25, size=5, cell_type_ids=[1, 2, 2, 2, 2, 1])

        self.create_arranged_cells(x_s=50, y_s=5, size=5, cell_type_ids=[1, 2, 2, 2, 2, 1])
        self.create_arranged_cells(x_s=60, y_s=40, size=5, cell_type_ids=[1, 2, 2, 2, 2, 1])
        self.create_arranged_cells(x_s=70, y_s=60, size=5, cell_type_ids=[1, 2, 2, 2, 2, 1])

        for cell in self.cell_list:
            print("cell id=", cell.id, " clusterId=", cell.clusterId)

    def create_arranged_cells(self, x_s, y_s, size, cell_type_ids=None):
        """
        this function creates vertically arranged cells.

        x_s, ys - coordinates of bottom_left corner of the cell arrangement
        size - size of the cell arrangement
        cell_type_ids - list of cell type ids

        """
        cluster_id = None
        for i, cell_type_id in enumerate(cell_type_ids):
            cell = self.new_cell(cell_type=cell_type_id)

            if i == 0:
                cluster_id = cell.clusterId
                self.leading_cells_ids.add(cell.id)
            else:
                # to make all cells created by this function, we must reassign clusterId
                # of all the cells created by this function except the first one
                # When the first cell gets created, it gets reassigned clusterId by
                # CompuCell3D and we will use this clusterId to assign it to all other cells created by this function
                self.reassign_cluster_id(cell=cell, cluster_id=cluster_id)
            self.cell_field[x_s : x_s + size, y_s + i * size : y_s + (i + 1) * size, 0] = cell

    def step(self, mcs):

        if mcs < 300:
            return

        if not mcs % 500:
            # randomize force applied to leading cell
            for cell in self.cell_list:
                if cell.id in self.leading_cells_ids:
                    cell.lambdaVecX = random.randint(-self.maxAbsLambdaX, self.maxAbsLambdaX)
                    cell.lambdaVecY = random.randint(-self.maxAbsLambdaX, self.maxAbsLambdaX)