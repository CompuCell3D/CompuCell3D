import cc3d.CompuCellSetup as CompuCellSetup
from .cellsort_2D_steppables import CellsortSteppable

CompuCellSetup.register_steppable(steppable=CellsortSteppable(frequency=1))

CompuCellSetup.run()

