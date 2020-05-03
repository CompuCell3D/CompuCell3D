from cc3d import CompuCellSetup

from cellsort_2D_hexSteppables import cellsort_2D_hexSteppable

CompuCellSetup.register_steppable(steppable=cellsort_2D_hexSteppable(frequency=1))

CompuCellSetup.run()
