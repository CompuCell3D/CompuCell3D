from cc3d.core.PySteppables import *

from math import *


class MomentOfInertiaPrinter(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.semi_minor_axis = 15
        self.semi_major_axis = 5

    def start(self):
        self.generate_ellipse(self.semi_minor_axis, self.semi_major_axis)

    def generate_ellipse(self, semi_minor_axis, semi_major_axis):
        cell = self.new_cell(cell_type=1)

        for x in range(self.dim.x):
            for y in range(self.dim.y):
                if (x - self.dim.x / 2.0) ** 2 / semi_minor_axis ** 2 + (
                        y - self.dim.y / 2.0) ** 2 / semi_major_axis ** 2 < 1:
                    self.cell_field[x, y, 0] = cell

    def step(self, mcs):
        for cell in self.cell_list:
            print("CELL ID=", cell.id, " CELL TYPE=", cell.type, " volume=", cell.volume, " lX=", cell.lX, " lY=",
                  cell.lY, " ecc=", cell.ecc)
            print("cell.iXX=", cell.iXX, " cell.iYY=", cell.iYY, " cell.iXY=", cell.iXY)
            radical = 0.5 * sqrt((cell.iXX - cell.iYY) * (cell.iXX - cell.iYY) + 4.0 * cell.iXY * cell.iXY)
            print("radical=", radical)
            l_min = 0.5 * (cell.iXX + cell.iYY) - radical
            l_max = 0.5 * (cell.iXX + cell.iYY) + radical
            # b=sqrt(cell.iXY**2+(l_max-cell.iXX)**2)
            # print "|b|=",b
            lSemiMajor = 2 * sqrt(l_max / cell.volume)
            lSemiMinor = 2 * sqrt(l_min / cell.volume)
            ecc = sqrt(1 - (lSemiMinor / lSemiMajor) ** 2)
            print("lSemiMajor=", lSemiMajor, " lSemiMinor=", lSemiMinor)
            print("cell.ecc=", cell.ecc, " ecc=", ecc)
            print("l_min=", l_min / cell.volume, " l_max=", l_max / cell.volume)

            # preferred way of accessing information about semiminor axes
            axes = self.momentOfInertiaPlugin.getSemiaxes(cell)
            print("minorAxis=", axes[0], " majorAxis=", axes[2], " medianAxis=", axes[1])
