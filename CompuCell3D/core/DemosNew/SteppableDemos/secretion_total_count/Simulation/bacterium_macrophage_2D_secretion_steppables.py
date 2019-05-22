from cc3d.core.PySteppables import *
from pathlib import Path


class SecretionSteppable(SecretionBasePy):
    def __init__(self, frequency=1):
        SecretionBasePy.__init__(self, frequency)

    def step(self, mcs):
        attr_secretor = self.get_field_secretor("ATTR")

        for cell in self.cellList:
            if cell.type == self.WALL:
                attr_secretor.secreteInsideCellAtBoundaryOnContactWith(cell, 300, [self.WALL])
                attr_secretor.secreteOutsideCellAtBoundaryOnContactWith(cell, 300, [self.MEDIUM])
                res = attr_secretor.secreteInsideCellTotalCount(cell, 300)
                print('secreted  ', res.tot_amount, ' inside cell')
                attr_secretor.secreteInsideCellAtBoundaryTotalCount(cell, 300)
                print('secreted  ', res.tot_amount, ' inside cell at the boundary')
                attr_secretor.secreteOutsideCellAtBoundary(cell, 500)
                attr_secretor.secreteInsideCellAtCOM(cell, 300)

                res = attr_secretor.uptakeInsideCellTotalCount(cell, 3, 0.1)
                print('Total uptake inside cell ', res.tot_amount)

                attr_secretor.uptakeInsideCellAtBoundaryOnContactWith(cell, 3, 0.1, [self.MEDIUM])
                attr_secretor.uptakeOutsideCellAtBoundaryOnContactWith(cell, 3, 0.1, [self.MEDIUM])

                res = attr_secretor.uptakeInsideCellAtBoundaryTotalCount(cell, 3, 0.1)
                print('Total uptake inside cell at the boundary ', res.tot_amount)
                attr_secretor.uptakeOutsideCellAtBoundary(cell, 3, 0.1)
                attr_secretor.uptakeInsideCellAtCOM(cell, 3, 0.1)

        output_dir = self.output_dir

        if output_dir is not None:
            output_path = Path(output_dir).joinpath('step_' + str(mcs).zfill(3) + '.dat')
            with open(output_path, 'w') as fout:

                attr_field = self.field.ATTR
                for x, y, z in self.every_pixel():
                    fout.write('{} {} {} {}\n'.format(x, y, z, attr_field[x, y, z]))
