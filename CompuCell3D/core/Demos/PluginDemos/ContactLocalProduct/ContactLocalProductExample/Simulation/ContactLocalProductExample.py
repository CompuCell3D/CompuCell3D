import sys
from os import environ
import string
sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup

sim,simthread = CompuCellSetup.getCoreSimulationObjects()

#Create extra player fields here or add attributes or Python plugins

CompuCellSetup.initializeSimulationObjects(sim,simthread)

#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()

from ContactLocalProductExampleModules import ContactLocalProductSteppable
clpSteppable=ContactLocalProductSteppable(sim)
typeContactEnergyTable={0:0.0 , 1:20, 2:30}
clpSteppable.setTypeContactEnergyTable(typeContactEnergyTable)
steppableRegistry.registerSteppable(clpSteppable)



CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

