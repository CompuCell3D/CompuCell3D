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

from cellsort_2D_steering_steppables import PottsSteering
pottsSteering=PottsSteering(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(pottsSteering)

from cellsort_2D_steering_steppables import ContactSteering
contactSteering=ContactSteering(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(contactSteering)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

