import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup

sim,simthread = CompuCellSetup.getCoreSimulationObjects()
import CompuCell

#Create extra player fields here or add attributes
CompuCellSetup.initializeSimulationObjects(sim,simthread)


#Add Python steppables here
from PySteppablesExamples import SteppableRegistry
steppableRegistry=SteppableRegistry()

from wetFoamSteppables import FlexCellInitializer
fci=FlexCellInitializer(_simulator=sim,_frequency=1)
fci.addCellTypeParameters(_type=1,_count=80,_targetVolume=25,_lambdaVolume=10.0)
fci.addCellTypeParameters(_type=2,_count=0,_targetVolume=5,_lambdaVolume=2.0)
fci.setFractionOfWater(0.25)
steppableRegistry.registerSteppable(fci)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

