import sys
from os import environ
from os import getcwd
import string
from PySteppablesExamples import SimulationFileStorage

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup


sim,simthread = CompuCellSetup.getCoreSimulationObjects()

#Create extra player fields here or add attributes

CompuCellSetup.initializeSimulationObjects(sim,simthread)

#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()

from CellMotilitySteppables import CellMotilitySteppable
cellMotilitySteppable=CellMotilitySteppable(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(cellMotilitySteppable)


CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

