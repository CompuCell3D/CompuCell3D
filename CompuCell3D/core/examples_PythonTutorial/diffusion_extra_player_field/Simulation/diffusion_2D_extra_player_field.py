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


#UNCOMMENTING NEXT LINE WILL KEEP EXTRA FIELD FROM BEING OUTPUT IN THE VTK FILE WHEN USIN runScript.bat                                                                
#CompuCellSetup.doNotOutputField("ExtraField")
                                                                
#Add Python steppables here
from PySteppablesExamples import SteppableRegistry
steppableRegistry=SteppableRegistry()

from diffusion_2D_steppables_player import ExtraFieldVisualizationSteppable
extraFieldVisualizationSteppable=ExtraFieldVisualizationSteppable(_simulator=sim,_frequency=10)
extraFieldVisualizationSteppable.setScalarField(extraPlayerField)
steppableRegistry.registerSteppable(extraFieldVisualizationSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

