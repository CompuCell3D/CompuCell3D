from cc3d import CompuCellSetup
from .CellInitializerSteppables import CellInitializer

CompuCellSetup.register_steppable(steppable=CellInitializer(frequency=100))

CompuCellSetup.run()

