from cc3d.core.PySteppables import *


class ElongationFlexSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        for cell in self.cell_list:
            if cell.type == 1:
                # cell , lambdaLength, targetLength
                self.lengthConstraintPlugin.setLengthConstraintData(cell, 20, 20)
                cell.connectivityOn = True

            elif cell.type == 2:
                # cell , lambdaLength, targetLength
                self.lengthConstraintPlugin.setLengthConstraintData(cell, 20, 30)
                cell.connectivityOn = True



