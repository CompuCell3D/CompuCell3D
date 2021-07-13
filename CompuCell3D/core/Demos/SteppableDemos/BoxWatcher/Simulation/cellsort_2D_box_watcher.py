from cc3d import CompuCellSetup

from cellsort_2D_box_watcherSteppables import cellsort_2D_box_watcherSteppable

CompuCellSetup.register_steppable(steppable=cellsort_2D_box_watcherSteppable(frequency=1))

CompuCellSetup.run()
