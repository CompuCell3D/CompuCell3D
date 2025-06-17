
from cc3d import CompuCellSetup
        

from SelfReinforcingSteppables import SelfReinforcingSteppable

CompuCellSetup.register_steppable(steppable=SelfReinforcingSteppable(frequency=1))


CompuCellSetup.run()
