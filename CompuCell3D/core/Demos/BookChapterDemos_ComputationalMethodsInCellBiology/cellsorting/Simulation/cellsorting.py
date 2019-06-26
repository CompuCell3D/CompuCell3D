from cc3d import CompuCellSetup
from .cellsortingSteppables import CellSortingSteppable

CompuCellSetup.register_steppable(steppable=CellSortingSteppable(frequency=100))

CompuCellSetup.run()