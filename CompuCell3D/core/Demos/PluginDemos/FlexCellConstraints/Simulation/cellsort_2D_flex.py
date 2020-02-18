from cc3d import CompuCellSetup

from cellsort_2D_flexSteppables import cellsort_2D_flexSteppable

CompuCellSetup.register_steppable(steppable=cellsort_2D_flexSteppable(frequency=1))

CompuCellSetup.run()
