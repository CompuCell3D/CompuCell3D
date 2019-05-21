from cc3d import CompuCellSetup
from .DynamicNumberOfProcessorsSteppables import DynamicNumberOfProcessorsSteppable

CompuCellSetup.register_steppable(steppable=DynamicNumberOfProcessorsSteppable(frequency=10))

CompuCellSetup.run()

#
# import sys
# from os import environ
# from os import getcwd
# import string
#
# sys.path.append(environ["PYTHON_MODULE_PATH"])
#
# import CompuCellSetup
#
# sim,simthread = CompuCellSetup.getCoreSimulationObjects()
#
# CompuCellSetup.initializeSimulationObjects(sim,simthread)
#
# #Add Python steppables here
# steppableRegistry=CompuCellSetup.getSteppableRegistry()
#
# from DynamicNumberOfProcessorsSteppables import DynamicNumberOfProcessorsSteppable
# steppableInstance=DynamicNumberOfProcessorsSteppable(sim,_frequency=10)
# steppableRegistry.registerSteppable(steppableInstance)
#
# CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)
