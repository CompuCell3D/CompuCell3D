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

from scientificHistBarPlotsSteppables import HistPlotSteppable
histPlotSteppable=HistPlotSteppable(_simulator=sim,_frequency=1)
steppableRegistry.registerSteppable(histPlotSteppable)

from scientificHistBarPlotsSteppables import BarPlotSteppable
barPlotSteppable=BarPlotSteppable(_simulator=sim,_frequency=1)
steppableRegistry.registerSteppable(barPlotSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

