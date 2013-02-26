

import sys
from os import environ
from os import getcwd
import string
## sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup

# Initialize core CompuCell3D simulation objects
sim, simthread = CompuCellSetup.getCoreSimulationObjects()
CompuCellSetup.initializeSimulationObjects(sim,simthread)

from TestCellDeathSteppables import *

# Create instances of required steppables
testCellDeathSteppable = TestCellDeathSteppable( sim )
mitosisSteppable = MitosisSteppable( sim )

# Add steppables to the steppable registry
steppableRegistry=CompuCellSetup.getSteppableRegistry()
steppableRegistry.registerSteppable( testCellDeathSteppable )
steppableRegistry.registerSteppable( mitosisSteppable )

CompuCellSetup.mainLoop(sim, simthread, steppableRegistry)



