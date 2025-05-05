
from cc3d import CompuCellSetup
        

from SchnackenbergCC3Dv4Steppables import SchnackenbergCC3Dv4Steppable

CompuCellSetup.register_steppable(steppable=SchnackenbergCC3Dv4Steppable(frequency=1))


CompuCellSetup.run()
