
from cc3d import CompuCellSetup
        

from GiererMeinhardtSteppables import GiererMeinhardtSteppable

CompuCellSetup.register_steppable(steppable=GiererMeinhardtSteppable(frequency=1))


CompuCellSetup.run()
