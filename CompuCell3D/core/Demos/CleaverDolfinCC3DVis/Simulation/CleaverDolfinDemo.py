import sys
from os import environ
import string
sys.path.append(environ["PYTHON_MODULE_PATH"])
from dolfin import *
import numpy,math
import CompuCellSetup
sim,simthread = CompuCellSetup.getCoreSimulationObjects()

CompuCellSetup.initializeSimulationObjects(sim,simthread)


#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()


from CleaverDolfinDemoSteppables import CleaverDolfinDemoSteppable
cdDemo=CleaverDolfinDemoSteppable(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(cdDemo)


CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)



