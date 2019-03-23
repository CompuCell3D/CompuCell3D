from cc3d import CompuCellSetup
from cellsort_2D_steppables_info_printer import InfoPrinterSteppable

sim, simthread = CompuCellSetup.getCoreSimulationObjects()

# Create extra player fields here or add attributes

CompuCellSetup.initializeSimulationObjects(sim, simthread)

# Add Python steppables here
steppableRegistry = CompuCellSetup.getSteppableRegistry()

infoPrinterSteppable = InfoPrinterSteppable(_simulator=sim, _frequency=10)
steppableRegistry.registerSteppable(infoPrinterSteppable)

CompuCellSetup.mainLoop(sim, simthread, steppableRegistry)
