

import sys
from os import environ
from os import getcwd
import string
## sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup

# Initialize core CompuCell3D simulation objects
sim, simthread = CompuCellSetup.getCoreSimulationObjects()
CompuCellSetup.initializeSimulationObjects(sim,simthread)

from OscillatingContactEnergiesSteppables import *

# Create instances of required steppables
oscillatingContactEnergiesSteppable = OscillatingContactEnergiesSteppable( sim )

# Add steppables to the steppable registry
steppableRegistry = CompuCellSetup.getSteppableRegistry()
steppableRegistry.registerSteppable( oscillatingContactEnergiesSteppable )

CompuCellSetup.mainLoop(sim, simthread, steppableRegistry)



