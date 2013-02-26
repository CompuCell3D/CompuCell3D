import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup
sim,simthread = CompuCellSetup.getCoreSimulationObjects()


import CompuCell #notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()

#Create extra player fields here or add attributes
CompuCellSetup.initializeSimulationObjects(sim,simthread)

                                                                
#Add Python steppables here
from PySteppablesExamples import SteppableRegistry
steppableRegistry=SteppableRegistry()

from ExtraFieldsSteppables import ExtraFieldVisualizationSteppable
extraFieldVisualizationSteppable=ExtraFieldVisualizationSteppable(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(extraFieldVisualizationSteppable)


from ExtraFieldsSteppables import IdFieldVisualizationSteppable
idFieldVisualizationSteppable=IdFieldVisualizationSteppable(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(idFieldVisualizationSteppable)

from ExtraFieldsSteppables import VectorFieldVisualizationSteppable
vectorFieldVisualizationSteppable=VectorFieldVisualizationSteppable(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(vectorFieldVisualizationSteppable)

from ExtraFieldsSteppables import VectorFieldCellLevelVisualizationSteppable
vectorFieldCellLevelVisualizationSteppable=VectorFieldCellLevelVisualizationSteppable(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(vectorFieldCellLevelVisualizationSteppable)


from ExtraFieldsSteppables import DiffusionFieldSteppable
instanceOfDiffusionFieldSteppable=DiffusionFieldSteppable(_simulator=sim,_frequency=1)
steppableRegistry.registerSteppable(instanceOfDiffusionFieldSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

