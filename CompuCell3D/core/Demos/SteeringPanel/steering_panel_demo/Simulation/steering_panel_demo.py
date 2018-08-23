import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup

sim, simthread = CompuCellSetup.getCoreSimulationObjects()

# Create extra player fields here or add attributes

CompuCellSetup.initializeSimulationObjects(sim, simthread)

# Add Python steppables here
steppableRegistry = CompuCellSetup.getSteppableRegistry()

# from cellsort_2D_steppables_info_printer import InfoPrinterSteppable
# infoPrinterSteppable=InfoPrinterSteppable(_simulator=sim,_frequency=10)
# steppableRegistry.registerSteppable(infoPrinterSteppable)

from steering_panel_demo_steppables import VolumeSteeringSteppable

volumeSteeringSteppable = VolumeSteeringSteppable(_simulator=sim, _frequency=1)
steppableRegistry.registerSteppable(volumeSteeringSteppable)

from steering_panel_demo_steppables import SurfaceSteeringSteppable

surfaceSteeringSteppable = SurfaceSteeringSteppable(_simulator=sim, _frequency=1)
steppableRegistry.registerSteppable(surfaceSteeringSteppable)

# from scientificPlotsSteppables import PlotSteppable2
# plotSteppable2=PlotSteppable2(_simulator=sim,_frequency=1)
# steppableRegistry.registerSteppable(plotSteppable2)


CompuCellSetup.mainLoop(sim, simthread, steppableRegistry)
