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

from diffusion_2D_steppables import ConcentrationFieldDumperSteppable
concentrationFieldDumperSteppable=ConcentrationFieldDumperSteppable(_simulator=sim,_frequency=100)
steppableRegistry.registerSteppable(concentrationFieldDumperSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

