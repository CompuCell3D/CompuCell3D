import cc3d.CompuCellSetup as CompuCellSetup
from .CellInitializerSteppables import CellInitializer

CompuCellSetup.register_steppable(steppable=CellInitializer(frequency=100))

CompuCellSetup.run()

