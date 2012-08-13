import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup

#CompuCellSetup.setSimulationXMLFileName("Simulation/FocalPointPlasticityCompartments.xml")

sim,simthread = CompuCellSetup.getCoreSimulationObjects()

#Create extra player fields here or add attributes

CompuCellSetup.initializeSimulationObjects(sim,simthread)

#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()

from FocalPointPlasticityCompartmentsSteppables import FocalPointPlasticityCompartmentsParams
focalPointPlasticityCompartmentsParams=FocalPointPlasticityCompartmentsParams(_simulator=sim,_frequency=10)
steppableRegistry.registerSteppable(focalPointPlasticityCompartmentsParams)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

