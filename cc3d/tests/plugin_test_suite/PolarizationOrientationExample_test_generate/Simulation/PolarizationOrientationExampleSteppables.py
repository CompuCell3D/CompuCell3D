from cc3d.core.PySteppables import *


class PolarizationOrientationExampleSteppable(SteppableBasePy):
    def __init__(self, frequency=100):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        for cell in self.cellList:
            self.polarizationVectorPlugin.setPolarizationVector(cell, 1, 1, 0)

            # uncomment this line if you want to use cell id - based lambdaOrientation.
            # Make sure you do not list any tags inside <Plugin Name="CellOrientation"> element in the xml file
            # self.cellOrientationPlugin.setLambdaCellOrientation(cell,50.0)
