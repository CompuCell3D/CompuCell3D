
from cc3d import CompuCellSetup
        

from ThomasCC3Dv4Steppables import ThomasCC3Dv4Steppable

CompuCellSetup.register_steppable(steppable=ThomasCC3Dv4Steppable(frequency=1))


CompuCellSetup.run()
