
from cc3d import CompuCellSetup
        

from ChemotaxisRunModelSteppables import ChemotaxisRunModelSteppable

CompuCellSetup.register_steppable(steppable=ChemotaxisRunModelSteppable(frequency=1))


CompuCellSetup.run()
