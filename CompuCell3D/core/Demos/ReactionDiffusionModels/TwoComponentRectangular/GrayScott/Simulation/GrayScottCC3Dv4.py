
from cc3d import CompuCellSetup
        

from GrayScottCC3Dv4Steppables import GrayScottCC3Dv4Steppable

CompuCellSetup.register_steppable(steppable=GrayScottCC3Dv4Steppable(frequency=1))


CompuCellSetup.run()
