from cc3d import CompuCellSetup
from .cellsort_2D_steppables_neighbor_tracker import NeighborTrackerPrinterSteppable

CompuCellSetup.register_steppable(steppable=NeighborTrackerPrinterSteppable(frequency=100))

CompuCellSetup.run()

