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

dim=sim.getPotts().getCellFieldG().getDim()
extraPlayerField=simthread.createFloatFieldPy(dim,"ExtraField") # initializing pressure Field - this location in the code is important this must be called before
                                                                ##preStartInit or otherwise field list will not be initialized properly

idField=simthread.createScalarFieldCellLevelPy("IdField") 

vectorField=simthread.createVectorFieldPy(dim,"VectorField") 

vectorCellLevelField=simthread.createVectorFieldCellLevelPy("VectorFieldCellLevel") 
                                                                
#Add Python steppables here
from PySteppablesExamples import SteppableRegistry
steppableRegistry=SteppableRegistry()

from ExtraFields_steppables import ExtraFieldVisualizationSteppable
extraFieldVisualizationSteppable=ExtraFieldVisualizationSteppable(_simulator=sim,_frequency=10)
extraFieldVisualizationSteppable.setScalarField(extraPlayerField)
steppableRegistry.registerSteppable(extraFieldVisualizationSteppable)


from ExtraFields_steppables import IdFieldVisualizationSteppable
idFieldVisualizationSteppable=IdFieldVisualizationSteppable(_simulator=sim,_frequency=10)
idFieldVisualizationSteppable.setScalarField(idField)
steppableRegistry.registerSteppable(idFieldVisualizationSteppable)

from ExtraFields_steppables import VectorFieldVisualizationSteppable
vectorFieldVisualizationSteppable=VectorFieldVisualizationSteppable(_simulator=sim,_frequency=10)
vectorFieldVisualizationSteppable.setVectorField(vectorField)
steppableRegistry.registerSteppable(vectorFieldVisualizationSteppable)

from ExtraFields_steppables import VectorFieldCellLevelVisualizationSteppable
vectorFieldCellLevelVisualizationSteppable=VectorFieldCellLevelVisualizationSteppable(_simulator=sim,_frequency=10)
vectorFieldCellLevelVisualizationSteppable.setVectorField(vectorCellLevelField)
steppableRegistry.registerSteppable(vectorFieldCellLevelVisualizationSteppable)



CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

