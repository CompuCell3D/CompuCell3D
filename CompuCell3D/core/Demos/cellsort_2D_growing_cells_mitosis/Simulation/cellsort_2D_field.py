import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])



import CompuCellSetup

CompuCellSetup.setSimulationXMLFileName("Simulation/cellsort_2D_field.xml")

sim,simthread = CompuCellSetup.getCoreSimulationObjects()

#add additional attributes
pyAttributeAdder,listAdder=CompuCellSetup.attachListToCells(sim)

CompuCellSetup.initializeSimulationObjects(sim,simthread)

import CompuCell #notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()
changeWatcherRegistry=CompuCellSetup.getChangeWatcherRegistry(sim)
stepperRegistry=CompuCellSetup.getStepperRegistry(sim)

from cellsort_2D_field_modules import CellsortMitosis
cellsortMitosis=CellsortMitosis(sim,changeWatcherRegistry,stepperRegistry)
cellsortMitosis.setDoublingVolume(50)

#changeWatcherRegistry.registerPyChangeWatcher(cellsortMitosis)
#stepperRegistry.registerPyStepper(cellsortMitosis)



#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()


from cellsort_2D_field_modules import VolumeConstraintSteppable
volumeConstraint=VolumeConstraintSteppable(sim)
steppableRegistry.registerSteppable(volumeConstraint)

from cellsort_2D_field_modules import MitosisDataPrinterSteppable
mitosisDataPrinterSteppable=MitosisDataPrinterSteppable(sim)
steppableRegistry.registerSteppable(mitosisDataPrinterSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)




