import sys
from os import environ
import string
sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup
CompuCellSetup.setSimulationXMLFileName("Simulation/elongationFlexTest.xml")
sim,simthread = CompuCellSetup.getCoreSimulationObjects()

CompuCellSetup.initializeSimulationObjects(sim,simthread)

from PySteppables import SteppableRegistry
steppableRegistry=SteppableRegistry()

from elongationFlexSteppables import ElongationFlexSteppable
elongationFlexSteppable=ElongationFlexSteppable(_simulator=sim,_frequency=50)
steppableRegistry.registerSteppable(elongationFlexSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

