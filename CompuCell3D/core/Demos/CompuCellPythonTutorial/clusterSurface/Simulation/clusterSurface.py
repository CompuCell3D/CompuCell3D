import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup

sim,simthread = CompuCellSetup.getCoreSimulationObjects()
CompuCellSetup.initializeSimulationObjects(sim,simthread)
import CompuCell #notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()

#Add Python steppables here
from PySteppablesExamples import SteppableRegistry
steppableRegistry=SteppableRegistry()

from clusterSurfaceSteppables import VolumeParamSteppable
volumeParamSteppable=VolumeParamSteppable(sim,10)
steppableRegistry.registerSteppable(volumeParamSteppable)

from clusterSurfaceSteppables import MitosisSteppableClusters
mitosisSteppable=MitosisSteppableClusters(sim,10)
steppableRegistry.registerSteppable(mitosisSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)



