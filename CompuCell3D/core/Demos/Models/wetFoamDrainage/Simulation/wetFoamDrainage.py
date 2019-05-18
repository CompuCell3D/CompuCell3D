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

from wetFoamDrainageSteppables import FlexCellInitializer
fci=FlexCellInitializer(_simulator=sim,_frequency=1)
fci.add_cell_type_parameters(cell_type=1, count=80, target_volume=25, lambda_volume=10.0)
fci.add_cell_type_parameters(cell_type=2, count=0, target_volume=5, lambda_volume=2.0)
fci.set_fraction_of_water(0.25)
steppableRegistry.registerSteppable(fci)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

