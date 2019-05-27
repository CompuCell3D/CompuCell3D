from cc3d import CompuCellSetup
from .scientificPlotsSteppables import ExtraPlotSteppable

CompuCellSetup.register_steppable(steppable=ExtraPlotSteppable(frequency=1))

CompuCellSetup.run()


# import sys
# from os import environ
# from os import getcwd
# import string
#
# sys.path.append(environ["PYTHON_MODULE_PATH"])
#
#
# import CompuCellSetup
#
#
# sim,simthread = CompuCellSetup.getCoreSimulationObjects()
#
# #Create extra player fields here or add attributes
#
# CompuCellSetup.initializeSimulationObjects(sim,simthread)
#
# #Add Python steppables here
# steppableRegistry=CompuCellSetup.getSteppableRegistry()
#
# # from cellsort_2D_steppables_info_printer import InfoPrinterSteppable
# # infoPrinterSteppable=InfoPrinterSteppable(_simulator=sim,_frequency=10)
# # steppableRegistry.registerSteppable(infoPrinterSteppable)
#
# from scientificPlotsSteppables import ExtraPlotSteppable
# extraPlotSteppable=ExtraPlotSteppable(_simulator=sim,_frequency=1)
# steppableRegistry.registerSteppable(extraPlotSteppable)
#
#
# CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)
#
