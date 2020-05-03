
from cc3d.core.PySteppables import *

from cc3d.CompuCellSetup import persistent_globals as pg


class ChemotaxisRunModelSteppable(SteppableBasePy):

    def __init__(self, frequency=1):

        SteppableBasePy.__init__(self,frequency)
        
        self.cell_OI = None
        self.man_lam = 1340.4239654411663  # Solution from optimization

    def start(self):
        """
        any code in the start function runs before MCS=0
        """
        self.cell_OI = self.new_cell(self.TYPE1)
        self.cell_field[47:53, 22:28, 0] = self.cell_OI
        
        cd = self.chemotaxisPlugin.addChemotaxisData(self.cell_OI, "Field1")
        # Get input data from outside world if available
        if pg.input_object is not None:
            cd.setLambda(pg.input_object)
        else:
            cd.setLambda(self.man_lam)
        cd.assignChemotactTowardsVectorTypes([self.MEDIUM])

    def step(self, mcs):
        pass

    def finish(self):
        pg.return_object = self.cell_OI.xCOM


        