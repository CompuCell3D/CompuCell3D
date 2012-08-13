import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup
sim,simthread = CompuCellSetup.getCoreSimulationObjects()
import CompuCell

#Create extra player fields here or add attributes
pyAttributeAdder,listAdder=CompuCellSetup.attachListToCells(sim)




CompuCellSetup.initializeSimulationObjects(sim,simthread)

dim=sim.getPotts().getCellFieldG().getDim()

pressureField=simthread.createFloatFieldPy(dim,"PressurePy") # initializing pressure Field - this location in the code is important this must be called before
                                                                ##preStartInit or otherwise field list will not be initialized properly


#Add Python steppables here
from pressureField_steppables import SteppableRegistry
steppableRegistry=SteppableRegistry()

from pressureField_steppables import TargetVolumeDrosoSteppable
targetVolumeDrosoSteppable=TargetVolumeDrosoSteppable()
targetVolumeDrosoSteppable.setInitialTargetVolume(25)
targetVolumeDrosoSteppable.setInitialLambdaVolume(2.0)
steppableRegistry.registerSteppable(targetVolumeDrosoSteppable)

from pressureField_steppables import BlobSimpleTypeInitializer
blobTypeInitializer=BlobSimpleTypeInitializer(sim)
steppableRegistry.registerSteppable(blobTypeInitializer)

from pressureField_steppables import ModifyAttribute
modifyAttribute=ModifyAttribute(sim)
steppableRegistry.registerSteppable(modifyAttribute)

from pressureField_steppables import CellKiller
killer=CellKiller(sim)
steppableRegistry.registerSteppable(killer)

from pressureField_steppables import PressureFieldVisualizationSteppable
pressureFieldVisualizationSteppable=PressureFieldVisualizationSteppable(_simulator=sim,_frequency=10)
pressureFieldVisualizationSteppable.setScalarField(pressureField)
steppableRegistry.registerSteppable(pressureFieldVisualizationSteppable)



CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)



