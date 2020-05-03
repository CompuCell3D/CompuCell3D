
from cc3d import CompuCellSetup
        

from cellsort_2D_statisticsSteppables import cellsort_2D_statisticsSteppable

CompuCellSetup.register_steppable(steppable=cellsort_2D_statisticsSteppable(frequency=1))


CompuCellSetup.run()
