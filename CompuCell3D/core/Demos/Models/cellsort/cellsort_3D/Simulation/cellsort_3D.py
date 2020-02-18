from cc3d import CompuCellSetup

from cellsort_3DSteppables import cellsort_3DSteppable

CompuCellSetup.register_steppable(steppable=cellsort_3DSteppable(frequency=1))

CompuCellSetup.run()
