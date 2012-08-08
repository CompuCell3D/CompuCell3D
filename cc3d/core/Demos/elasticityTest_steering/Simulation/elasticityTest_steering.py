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

from elasticityTest_steering_steppables import ElasticitySteering
elasticitySteering=ElasticitySteering(sim,100)
steppableRegistry.registerSteppable(elasticitySteering)


CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

