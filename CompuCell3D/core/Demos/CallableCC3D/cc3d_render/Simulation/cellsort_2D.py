from cc3d import CompuCellSetup
from .cellsort_2D_steppables import CellsortSteppable

CompuCellSetup.register_steppable(steppable=CellsortSteppable(frequency=1))

CompuCellSetup.run()
