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


from elongationTest_steering_steppables import LengthConstraintSteering
lengthConstraintSteering=LengthConstraintSteering(_simulator=sim,_frequency=100)
steppableRegistry.registerSteppable(lengthConstraintSteering)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

