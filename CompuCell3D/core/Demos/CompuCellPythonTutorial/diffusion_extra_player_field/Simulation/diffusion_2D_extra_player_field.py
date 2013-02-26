import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup

sim,simthread = CompuCellSetup.getCoreSimulationObjects()


import CompuCell #notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()


CompuCellSetup.initializeSimulationObjects(sim,simthread)
                                                                
#Add Python steppables here
from PySteppablesExamples import SteppableRegistry
steppableRegistry=SteppableRegistry()

from diffusion_2D_steppables_player import ExtraFieldVisualizationSteppable
extraFieldVisualizationSteppable=ExtraFieldVisualizationSteppable(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(extraFieldVisualizationSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

