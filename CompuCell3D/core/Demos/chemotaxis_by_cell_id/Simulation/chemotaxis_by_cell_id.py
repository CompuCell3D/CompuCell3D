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


from chemotaxis_by_cell_id_steppables import ChemotaxisSteering
chemotaxisSteering=ChemotaxisSteering(_simulator=sim,_frequency=100)
steppableRegistry.registerSteppable(chemotaxisSteering)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

