import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup

#CompuCellSetup.setSimulationXMLFileName("Simulation/FocalPointPlasticity.xml")

sim,simthread = CompuCellSetup.getCoreSimulationObjects()

#Create extra player fields here or add attributes

CompuCellSetup.initializeSimulationObjects(sim,simthread)

#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()

from MomentOfInertiaSteppables import MomentOfInertiaPrinter
momentOfInertiaPrinter=MomentOfInertiaPrinter(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(momentOfInertiaPrinter)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

