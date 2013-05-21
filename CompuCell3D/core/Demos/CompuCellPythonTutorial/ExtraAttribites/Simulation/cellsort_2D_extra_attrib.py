import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup



sim,simthread = CompuCellSetup.getCoreSimulationObjects()

#Create extra player fields here or add attributes

CompuCellSetup.initializeSimulationObjects(sim,simthread)

#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()

#here we will add ExtraAttributeCellsort steppable
from cellsort_2D_steppables_extra_attrib import ExtraAttributeCellsort
extraAttributeCellsort=ExtraAttributeCellsort(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(extraAttributeCellsort)


from cellsort_2D_steppables_extra_attrib import TypeSwitcherSteppable
typeSwitcherSteppable=TypeSwitcherSteppable(sim,100)
steppableRegistry.registerSteppable(typeSwitcherSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)


