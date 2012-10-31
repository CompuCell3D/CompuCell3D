import sys
from os import environ
from os import getcwd
import string
from PySteppablesExamples import SimulationFileStorage

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup


sim,simthread = CompuCellSetup.getCoreSimulationObjects()

# Create extra player fields here or add attributes

CompuCellSetup.initializeSimulationObjects(sim, simthread)



# Add Python steppables here
steppableRegistry = CompuCellSetup.getSteppableRegistry()

from helpfile_steppables_CellDraw import HelpfileCellDrawSteppable

theHelpfileCellDrawSteppable = HelpfileCellDrawSteppable(_simulator=sim, _frequency=100)

steppableRegistry.registerSteppable( theHelpfileCellDrawSteppable )

# sfs=SimulationFileStorage(_simulator=sim,_frequency=10)
# sfs.addFileNameToStore("examples_PythonTutorial/cellsort_2D_info_printer/cellsort_2D.xml")
# sfs.addFileNameToStore("examples_PythonTutorial/cellsort_2D_info_printer/cellsort_2D_info_printer.py")
# sfs.addFileNameToStore("examples_PythonTutorial/cellsort_2D_info_printer/cellsort_2D_steppables_info_printer.py")
# steppableRegistry.registerSteppable(sfs)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

