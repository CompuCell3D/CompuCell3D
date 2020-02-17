
from cc3d.core.PySteppables import *

class BMFormation2ComponentDemoSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """

    def step(self,mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """
        # Put back all first component during initialization, to let cells settle
        if mcs < -100:
            pt = CompuCell.Point3D()
            for x, y, z in self.every_pixel():
                if self.cell_field[x, y, z]:
                    continue
                
                pt.x = x
                pt.y = y
                pt.z = z
                self.NCMaterialsPlugin.setMediumNCMaterialQuantityByIndex(pt, 0, 1.0)
                self.NCMaterialsPlugin.setMediumNCMaterialQuantityByIndex(pt, 1, 0.0)

    def finish(self):
        """
        Finish Function is called after the last MCS
        """


        