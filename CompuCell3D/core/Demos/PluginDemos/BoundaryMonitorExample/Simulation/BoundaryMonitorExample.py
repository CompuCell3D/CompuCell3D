import sys
from os import environ
import string
sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup

sim,simthread = CompuCellSetup.getCoreSimulationObjects()

CompuCellSetup.initializeSimulationObjects(sim,simthread)


#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()


from BoundaryMonitorExampleSteppables import BoundaryMonitorSteppable
boundaryMonitor=BoundaryMonitorSteppable(_simulator=sim,_frequency=1)
steppableRegistry.registerSteppable(boundaryMonitor)


CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)



