from cc3d import CompuCellSetup

from cellsort_2D_boundarySteppables import cellsort_2D_boundarySteppable

CompuCellSetup.register_steppable(steppable=cellsort_2D_boundarySteppable(frequency=1))

CompuCellSetup.run()
