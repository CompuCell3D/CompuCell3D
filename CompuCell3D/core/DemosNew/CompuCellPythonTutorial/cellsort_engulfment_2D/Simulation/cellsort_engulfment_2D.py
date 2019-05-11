from cc3d import CompuCellSetup
from .cellsort_engulfment_2D_steppables import CellInitializer

CompuCellSetup.register_steppable(steppable=CellInitializer(frequency=1))

CompuCellSetup.run()
