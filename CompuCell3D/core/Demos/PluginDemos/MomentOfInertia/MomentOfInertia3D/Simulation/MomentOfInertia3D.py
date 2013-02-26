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

from MomentOfInertia3DSteppables import MomentOfInertiaPrinter3D
momentOfInertiaPrinter3D=MomentOfInertiaPrinter3D(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(momentOfInertiaPrinter3D)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

