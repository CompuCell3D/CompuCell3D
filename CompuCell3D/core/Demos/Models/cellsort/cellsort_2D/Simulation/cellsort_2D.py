from cc3d import CompuCellSetup

from cellsort_2DSteppables import cellsort_2DSteppable

CompuCellSetup.register_steppable(steppable=cellsort_2DSteppable(frequency=1))

CompuCellSetup.run()
