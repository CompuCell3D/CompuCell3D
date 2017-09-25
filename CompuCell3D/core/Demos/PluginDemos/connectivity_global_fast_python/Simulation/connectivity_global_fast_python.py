import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup

sim,simthread = CompuCellSetup.getCoreSimulationObjects()
                        
CompuCellSetup.initializeSimulationObjects(sim,simthread)
steppableRegistry=CompuCellSetup.getSteppableRegistry()


from connectivity_global_fast_python_steppables import ConnectivitySteppable
instanceOfConnectivitySteppable=ConnectivitySteppable(_simulator=sim,_frequency=1)
steppableRegistry.registerSteppable(instanceOfConnectivitySteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)
        

