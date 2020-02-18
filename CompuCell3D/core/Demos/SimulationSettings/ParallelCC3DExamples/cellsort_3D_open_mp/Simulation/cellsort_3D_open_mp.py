from cc3d import CompuCellSetup

from cellsort_3D_open_mpSteppables import cellsort_3D_open_mpSteppable

CompuCellSetup.register_steppable(steppable=cellsort_3D_open_mpSteppable(frequency=1))

CompuCellSetup.run()
