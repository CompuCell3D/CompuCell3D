from cc3d.cpp import CompuCell
from cc3d.core.PySteppables import *


class BoundaryMonitorSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.pixel_assigned = False
        self.medium_cell = None
        self.boundary_array = None

    def start(self):

        self.medium_cell = CompuCell.getMediumCell()
        self.boundary_array = self.boundaryMonitorPlugin.getBoundaryArray()
        print('self.boundaryArray=', self.boundary_array)
        print('dir(self.boundaryArray)=', dir(self.boundary_array))

    def step(self, mcs):
        for cell in self.cell_list:
            if cell.type == 3:
                pt = CompuCell.Point3D()
                for x in range(9, 17):
                    for y in range(9, 17):
                        pt.x = x
                        pt.y = y
                        if int(self.boundary_array.get(pt)):
                            print('pt=', pt, ' boundary=', int(self.boundary_array.get(pt)))
            if not self.pixel_assigned:
                pt = CompuCell.Point3D(12, 12, 0)
                self.cell_field.set(pt, self.medium_cell)
                self.pixel_assigned = True
            if mcs == 3:
                self.cell_field[12, 12, 0] = cell
                print('REASSIGNMNET COMPLETED')

            if mcs == 4:
                self.cell_field[12, 10, 0] = self.medium_cell

            if mcs == 5:
                self.cell_field[12, 11, 0] = self.medium_cell

            break
