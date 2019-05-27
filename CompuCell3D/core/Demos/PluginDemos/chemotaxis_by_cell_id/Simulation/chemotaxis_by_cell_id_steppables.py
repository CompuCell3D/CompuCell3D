from cc3d.core.PySteppables import *


class ChemotaxisSteering(SteppableBasePy):
    def __init__(self, frequency=100):
        SteppableBasePy.__init__(self, frequency)

    def start(self):

        for cell in self.cell_list:
            if cell.type == self.MACROPHAGE:
                cd = self.chemotaxisPlugin.addChemotaxisData(cell, "ATTR")
                cd.setLambda(30.0)
                cd.assignChemotactTowardsVectorTypes([self.MEDIUM, self.BACTERIUM])
                break

    def step(self, mcs):
        if mcs > 100 and not mcs % 100:
            for cell in self.cell_list:
                if cell.type == self.MACROPHAGE:

                    cd = self.chemotaxisPlugin.getChemotaxisData(cell, "ATTR")
                    if cd:
                        lm = cd.getLambda() - 3
                        cd.setLambda(lm)
                    break
