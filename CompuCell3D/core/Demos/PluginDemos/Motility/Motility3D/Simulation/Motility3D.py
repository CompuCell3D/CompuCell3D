
from cc3d import CompuCellSetup
        

from Motility3DSteppables import Motility3DSteppable

CompuCellSetup.register_steppable(steppable=Motility3DSteppable(frequency=1))


CompuCellSetup.run()
