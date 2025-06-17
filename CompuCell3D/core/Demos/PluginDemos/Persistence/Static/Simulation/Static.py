
from cc3d import CompuCellSetup
        

from StaticSteppables import StaticSteppable

CompuCellSetup.register_steppable(steppable=StaticSteppable(frequency=1))


CompuCellSetup.run()
