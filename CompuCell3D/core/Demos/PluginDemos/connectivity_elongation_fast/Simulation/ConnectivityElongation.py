import sys
from os import environ
import string
sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup

sim,simthread = CompuCellSetup.getCoreSimulationObjects()

CompuCellSetup.initializeSimulationObjects(sim,simthread)

from PySteppables import SteppableRegistry
steppableRegistry=SteppableRegistry()

from ConnectivityElongationSteppable import ConnectivityElongationSteppable
connectivityElongationSteppable=ConnectivityElongationSteppable(_simulator=sim,_frequency=50)
steppableRegistry.registerSteppable(connectivityElongationSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

