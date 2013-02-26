import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup
sim,simthread = CompuCellSetup.getCoreSimulationObjects()

import CompuCell #notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()

#Create extra player fields here or add attributes
CompuCellSetup.initializeSimulationObjects(sim,simthread)

#Add Python steppables here
from PySteppablesExamples import SteppableRegistry
steppableRegistry=SteppableRegistry()

from FieldSecretionSteppables import FieldSecretionSteppable
fieldSecretionSteppable=FieldSecretionSteppable(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(fieldSecretionSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

