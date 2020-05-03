
from cc3d import CompuCellSetup
        

from cellsort_2D_statistics_spin_flipSteppables import cellsort_2D_statistics_spin_flipSteppable

CompuCellSetup.register_steppable(steppable=cellsort_2D_statistics_spin_flipSteppable(frequency=1))


CompuCellSetup.run()
