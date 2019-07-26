from cc3d import CompuCellSetup
from .cellsort_2D_steppables_info_printer import InfoPrinterSteppable

CompuCellSetup.register_steppable(steppable=InfoPrinterSteppable(frequency=1))

CompuCellSetup.run()
