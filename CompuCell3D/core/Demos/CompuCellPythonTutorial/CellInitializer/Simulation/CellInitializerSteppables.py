from cc3d.core.PySteppables import *
from math import pi
from os.path import dirname, join, abspath


class CellInitializer(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.medium_cell = None

    def start(self):

        self.medium_cell = self.cell_field[0, 0, 0]

        simulation_dir = dirname(abspath(__file__))
        file_full_name = join(simulation_dir, 'CellSurfaceCircularFactors.txt')
        # normalizing path
        file_full_name = abspath(file_full_name)

        file_handle = open(file_full_name, "w")

        for radius in range(1, 50):
            cell_surface, cell_volume = self.draw_circular_cell(radius)
            cell_surface_manual_calculation = self.calculate_htbl()
            file_handle.write(
                "%f %f %f %f %f\n" % (radius, 2 * pi * radius, cell_surface, pi * radius ** 2, cell_volume))
            print("radius=", radius, " surface theor=", 2 * pi * radius, " cell_surface=", cell_surface,
                  "cell_surface_manual_calculation=", cell_surface_manual_calculation)

        file_handle.close()

    def draw_circular_cell(self, _radius):

        x_center = self.dim.x / 2
        y_center = self.dim.y / 2
        z_center = self.dim.z / 2

        # assigning medium to all lattice points
        self.cell_field[:, :, :] = self.medium_cell

        # initializing large circular cell in the middle of the lattice
        cell = self.potts.createCell()
        cell.type = 1
        for x in range(self.dim.x):
            for y in range(self.dim.y):
                for z in range(self.dim.z):

                    if ((x - x_center) ** 2 + (y - y_center) ** 2) < _radius ** 2:
                        self.cell_field[x, y, z] = cell

        return cell.surface, cell.volume

    def calculate_htbl(self):

        cell_surface_manual_calculation = 0
        for x in range(self.dim.x):
            for y in range(self.dim.y):
                for z in range(self.dim.z):

                    cell = self.cell_field[x, y, z]
                    pt = CompuCell.Point3D(x, y, z)

                    if cell:
                        for pixel_neighbor in self.get_pixel_neighbors_based_on_neighbor_order(pt, 1):
                            # break
                            # continue
                            n_cell = self.cell_field[pixel_neighbor.pt.x, pixel_neighbor.pt.y, pixel_neighbor.pt.z]
                            if CompuCell.areCellsDifferent(n_cell, cell):
                                cell_surface_manual_calculation += 1

        return cell_surface_manual_calculation

    def step(self, mcs):
        pass

    def output_field(self, field_name, file_name):
        field = CompuCell.getConcentrationField(self.simulator, field_name)
        if field:
            try:
                file_handle = open(file_name, "w")
            except IOError:
                print("Could not open file ", file_name, " for writing. Check if you have necessary permissions")
                return

            print("dim.x=", self.dim.x)
            for i in range(self.dim.x):
                for j in range(self.dim.y):
                    for k in range(self.dim.z):
                        file_handle.write("%d\t%d\t%d\t%f\n" % (i, j, k, field[i, j, k]))
