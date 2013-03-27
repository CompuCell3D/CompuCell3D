import sys
from os import environ
import string
sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup

sim,simthread = CompuCellSetup.getCoreSimulationObjects()

CompuCellSetup.initializeSimulationObjects(sim,simthread)


#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()


from BoundaryPixelTrackerExampleSteppables import BoundaryPixelTrackerSteppable
boundaryPixelTracker=BoundaryPixelTrackerSteppable(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(boundaryPixelTracker)


CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)



