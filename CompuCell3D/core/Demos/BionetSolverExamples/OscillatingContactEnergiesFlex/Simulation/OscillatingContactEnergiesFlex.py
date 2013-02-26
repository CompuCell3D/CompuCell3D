

import sys
from os import environ
from os import getcwd
import string
## sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup

# Initialize core CompuCell3D simulation objects
sim, simthread = CompuCellSetup.getCoreSimulationObjects()
#Create extra player fields here or add attributes
pyAttributeAdder,listAdder=CompuCellSetup.attachDictionaryToCells(sim)

CompuCellSetup.initializeSimulationObjects(sim,simthread)

from OscillatingContactEnergiesFlexSteppables import *

# Create instances of required steppables
oscillatingContactEnergiesSteppable = OscillatingContactEnergiesSteppable( sim )

# Add steppables to the steppable registry
steppableRegistry = CompuCellSetup.getSteppableRegistry()
steppableRegistry.registerSteppable( oscillatingContactEnergiesSteppable )

CompuCellSetup.mainLoop(sim, simthread, steppableRegistry)



