from cc3d.core.PySteppables import *


class ChemotaxisCompartmentSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        cell_a = self.new_cell(self.A)
        cell_b = self.new_cell(self.B)
        cell_c = self.new_cell(self.C)

        self.cellField[60:80, 50:60, 0] = cell_a
        self.cellField[40:60, 50:60, 0] = cell_b
        self.cellField[20:40, 50:60, 0] = cell_c

        self.reassign_cluster_id(cell_b, 1)
        self.reassign_cluster_id(cell_c, 1)

        fgf = self.field.FGF

        for x in range(20, 75, 1):
            fgf[x, :, :] = x

    def step(self, mcs):
        for cell in self.cell_list:
            print('cell {cell_id} cluster_id={cluster_id}  '.format(cell_id=cell.id, cluster_id=cell.clusterId))
