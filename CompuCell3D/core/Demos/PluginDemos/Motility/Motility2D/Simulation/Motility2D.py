
from cc3d import CompuCellSetup
        

from Motility2DSteppables import Motility2DSteppable

CompuCellSetup.register_steppable(steppable=Motility2DSteppable(frequency=1))


CompuCellSetup.run()
