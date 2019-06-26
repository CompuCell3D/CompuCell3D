import cc3d.CompuCellSetup as CompuCellSetup
from .CellSortingSteppables import CellSortingSteppable


CompuCellSetup.register_steppable(steppable=CellSortingSteppable(frequency=1))
CompuCellSetup.run()
        

        
        