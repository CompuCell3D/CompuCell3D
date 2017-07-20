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


# from scientificPlotsSteppables import ExtraPlotSteppable
# extraPlotSteppable=ExtraPlotSteppable(_simulator=sim,_frequency=1)
# steppableRegistry.registerSteppable(extraPlotSteppable)


from scientificPlotsSteppables import ExtraMultiPlotSteppable
extraMultiPlotSteppable=ExtraMultiPlotSteppable(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(extraMultiPlotSteppable)


CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

