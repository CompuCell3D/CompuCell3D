import sys
from os import environ
import string
sys.path.append(environ["PYTHON_MODULE_PATH"])
   
import CompuCellSetup
sim,simthread = CompuCellSetup.getCoreSimulationObjects()

#Create extra player fields here or add attributes

CompuCellSetup.initializeSimulationObjects(sim,simthread)

#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()

from ContactMultiCadSteppables import ContactMultiCadSteppable
cmcSteppable=ContactMultiCadSteppable(sim)
typeContactEnergyTable={0:0.0 , 1:20, 2:30}
cmcSteppable.setTypeContactEnergyTable(typeContactEnergyTable)
steppableRegistry.registerSteppable(cmcSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

