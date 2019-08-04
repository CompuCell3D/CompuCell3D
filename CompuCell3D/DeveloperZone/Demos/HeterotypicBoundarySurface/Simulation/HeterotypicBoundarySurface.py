
from cc3d import CompuCellSetup
        

from HeterotypicBoundarySurfaceSteppables import HeterotypicBoundarySurfaceSteppable

CompuCellSetup.register_steppable(steppable=HeterotypicBoundarySurfaceSteppable(frequency=1))


CompuCellSetup.run()
