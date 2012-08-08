import sys
from os import environ
import string
sys.path.append(environ["PYTHON_MODULE_PATH"])
   
import CompuCellSetup
CompuCellSetup.setSimulationXMLFileName("Simulation/vectorPlot.xml")
sim,simthread = CompuCellSetup.getCoreSimulationObjects()

CompuCellSetup.initializeSimulationObjects(sim,simthread)

#Create extra player fields here or add attributes
vectorField=simthread.createVectorFieldCellLevelPy("vector_field");# initializing vector Field - 
                                                                           #this location in the code is important this must be called before
                                                                           #preStartInit or otherwise field list will not be initialized properly




#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()

from vectorPlotSteppables import VectorFieldPlotTestSteppable
vPlotTest=VectorFieldPlotTestSteppable(_simulator=sim,_frequency=10)
vPlotTest.setVectorField(vectorField)

steppableRegistry.registerSteppable(vPlotTest)


CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)



