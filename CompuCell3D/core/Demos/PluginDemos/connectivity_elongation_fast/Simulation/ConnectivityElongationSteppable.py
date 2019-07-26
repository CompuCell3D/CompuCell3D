from cc3d.core.PySteppables import *


class ConnectivityElongationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        for cell in self.cell_list:
            if cell.type == 1:
                cell.connectivityOn = True

            elif cell.type == 2:
                cell.connectivityOn = True

    def step(self, mcs):
        for cell in self.cellList:
            if cell.type == 1:
                # cell , lambdaLength, targetLength
                self.lengthConstraintPlugin.setLengthConstraintData(cell, 20, 20)

            elif cell.type == 2:
                # cell , lambdaLength, targetLength
                self.lengthConstraintPlugin.setLengthConstraintData(cell, 20, 30)

                print("targetLength=", self.lengthConstraintPlugin.getTargetLength(cell), " lambdaLength=",
                      self.lengthConstraintPlugin.getLambdaLength(cell))
