import cc3d.CompuCellSetup as CompuCellSetup
from .BuildWall3DSteppables import BuildWall3DSteppable

CompuCellSetup.register_steppable(steppable=BuildWall3DSteppable(frequency=1))

CompuCellSetup.run()