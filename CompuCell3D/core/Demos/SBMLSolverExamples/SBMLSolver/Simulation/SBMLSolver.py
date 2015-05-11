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
        
from SBMLSolverSteppables import SBMLSolverSteppable
steppableInstance=SBMLSolverSteppable(sim,_frequency=1)
steppableRegistry.registerSteppable(steppableInstance)
        
CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)
