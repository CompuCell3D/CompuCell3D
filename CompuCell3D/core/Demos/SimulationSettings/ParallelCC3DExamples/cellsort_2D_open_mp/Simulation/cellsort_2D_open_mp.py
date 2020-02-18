from cc3d import CompuCellSetup

from cellsort_2D_open_mpSteppables import cellsort_2D_open_mpSteppable

CompuCellSetup.register_steppable(steppable=cellsort_2D_open_mpSteppable(frequency=1))

CompuCellSetup.run()
