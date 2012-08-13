import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup

sim,simthread = CompuCellSetup.getCoreSimulationObjects()


import CompuCell #notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()

#Create extra player fields here or add attributes or plugins
energyFunctionRegistry=CompuCellSetup.getEnergyFunctionRegistry(sim)

from cellsort_2D_plugins_with_py_plugin import VolumeEnergyFunctionPlugin
volumeEnergy=VolumeEnergyFunctionPlugin(energyFunctionRegistry)
volumeEnergy.setParams(2.0,25.0)

energyFunctionRegistry.registerPyEnergyFunction(volumeEnergy)

CompuCellSetup.initializeSimulationObjects(sim,simthread)


#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)



