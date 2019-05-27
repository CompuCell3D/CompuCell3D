from cc3d.core.PySteppables import *


class SecretionSteppable(SecretionBasePy):
    def __init__(self, frequency=1):
        SecretionBasePy.__init__(self, frequency)

    def step(self, mcs):
        attr_secretor = self.get_field_secretor("ATTR")
        for cell in self.cell_list:
            if cell.type == self.WALL:
                attr_secretor.secreteInsideCellAtBoundaryOnContactWith(cell, 300, [self.WALL])
                attr_secretor.secreteOutsideCellAtBoundaryOnContactWith(cell, 300, [self.MEDIUM])
                attr_secretor.secreteInsideCell(cell, 300)
                attr_secretor.secreteInsideCellAtBoundary(cell, 300)
                attr_secretor.secreteOutsideCellAtBoundary(cell, 500)
                attr_secretor.secreteInsideCellAtCOM(cell, 300)
