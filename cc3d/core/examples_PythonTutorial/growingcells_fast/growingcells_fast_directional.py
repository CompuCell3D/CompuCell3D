import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup
CompuCellSetup.setSimulationXMLFileName("examples_PythonTutorial/growingcells_fast/growingcells_fast_directional.xml")

sim,simthread = CompuCellSetup.getCoreSimulationObjects()
CompuCellSetup.initializeSimulationObjects(sim,simthread)
import CompuCell #notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()

from growingcells_fast_directional_plugins import MitosisPyPlugin

changeWatcherRegistry=CompuCellSetup.getChangeWatcherRegistry(sim)

stepperRegistry=CompuCellSetup.getStepperRegistry(sim)

mitPy=MitosisPyPlugin(sim,changeWatcherRegistry,stepperRegistry)
mitPy.setDoublingVolume(50)


#Add Python steppables here
from PySteppablesExamples import SteppableRegistry
steppableRegistry=SteppableRegistry()

from growingcells_fast_directional_steppables import VolumeParamSteppable
volumeParamSteppable=VolumeParamSteppable(sim,10)
steppableRegistry.registerSteppable(volumeParamSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)



