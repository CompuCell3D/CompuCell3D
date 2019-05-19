from cc3d.core.PySteppables import *


class MomentOfInertiaPrinter3D(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.semi_minor_axis = 15
        self.semi_median_axis = 8
        self.semi_major_axis = 5

    def start(self):
        self.generate_ellipsoid(self.semi_minor_axis, self.semi_median_axis, self.semi_major_axis)

    def generate_ellipsoid(self, semi_minor_axis, semi_median_axis, semi_major_axis):
        cell = self.new_cell(cell_type=1)

        for x, y, z in self.every_pixel():
            if (x - self.dim.x / 2.0) ** 2 / semi_minor_axis ** 2 + (
                    y - self.dim.y / 2.0) ** 2 / semi_major_axis ** 2 + (
                    z - self.dim.z / 2.0) ** 2 / semi_median_axis ** 2 < 1:
                self.cell_field[x, y, z] = cell

    def step(self, mcs):
        for cell in self.cellList:
            # preferred way of accessing information about semiminor axes
            axes = self.momentOfInertiaPlugin.getSemiaxes(cell)
            print("minorAxis=", axes[0], " majorAxis=", axes[2], " medianAxis=", axes[1])
