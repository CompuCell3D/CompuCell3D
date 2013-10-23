import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup


sim,simthread = CompuCellSetup.getCoreSimulationObjects()

#Create extra player fields here or add attributes

CompuCellSetup.initializeSimulationObjects(sim,simthread)

#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()

from cellsort_2D_steppables_neighbor_tracker import NeighborTrackerPrinterSteppable
neighborTrackerPrinterSteppable=NeighborTrackerPrinterSteppable(sim,100)
steppableRegistry.registerSteppable(neighborTrackerPrinterSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)


