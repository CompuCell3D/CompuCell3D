import sys
from os import environ
from os import getcwd
import string
sys.path.append(environ["PYTHON_MODULE_PATH"])
import CompuCellSetup
sim,simthread = CompuCellSetup.getCoreSimulationObjects()     
CompuCellSetup.initializeSimulationObjects(sim,simthread)
        
#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()

import CompuCell
OrientedGrowthPlugin = CompuCell.getOrientedGrowthPlugin()  

from OrientedGrowthDemoSteppables import GrowthSteppable
GrowthSteppableInstance=GrowthSteppable(sim,_frequency=10)
steppableRegistry.registerSteppable(GrowthSteppableInstance)
  
from OrientedGrowthDemoSteppables import OrientedConstraintSteppable
OrientedConstraintSteppableInstance=OrientedConstraintSteppable(sim,_frequency=1,_OGPlugin=OrientedGrowthPlugin)
steppableRegistry.registerSteppable(OrientedConstraintSteppableInstance)
        
CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)