import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup
sim,simthread = CompuCellSetup.getCoreSimulationObjects()
import CompuCell


CompuCellSetup.initializeSimulationObjects(sim,simthread)

#Add Python steppables here
from pressureFieldSteppables import SteppableRegistry
steppableRegistry=SteppableRegistry()

from pressureFieldSteppables import TargetVolumeDrosoSteppable
targetVolumeDrosoSteppable=TargetVolumeDrosoSteppable(_simulator=sim)
targetVolumeDrosoSteppable.setInitialTargetVolume(25)
targetVolumeDrosoSteppable.setInitialLambdaVolume(2.0)
steppableRegistry.registerSteppable(targetVolumeDrosoSteppable)


from pressureFieldSteppables import CellKiller
killer=CellKiller(sim)
steppableRegistry.registerSteppable(killer)

from pressureFieldSteppables import PressureFieldVisualizationSteppable
pressureFieldVisualizationSteppable=PressureFieldVisualizationSteppable(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(pressureFieldVisualizationSteppable)



CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)



