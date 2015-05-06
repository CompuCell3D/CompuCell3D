import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup

sim,simthread = CompuCellSetup.getCoreSimulationObjects()
            
CompuCellSetup.initializeSimulationObjects(sim,simthread)
# Definitions of additional Python-managed fields go here
        
#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()

from cellsort_engulfment_2D_steppables import CellInitializer
instanceOfCellInitializer=CellInitializer(_simulator=sim,_frequency=1)
steppableRegistry.registerSteppable(instanceOfCellInitializer)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)
        
        