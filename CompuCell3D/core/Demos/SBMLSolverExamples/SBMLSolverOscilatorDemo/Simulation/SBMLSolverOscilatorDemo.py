import cc3d.CompuCellSetup as CompuCellSetup
from .SBMLSolverOscilatorDemoSteppables import SBMLSolverOscilatorDemoSteppable

CompuCellSetup.register_steppable(steppable=SBMLSolverOscilatorDemoSteppable(frequency=1))

CompuCellSetup.run()


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
# #Add Python steppables here
# steppableRegistry=CompuCellSetup.getSteppableRegistry()
#
# from SBMLSolverOscilatorDemoSteppables import SBMLSolverOscilatorDemoSteppable
# steppableInstance=SBMLSolverOscilatorDemoSteppable(sim,_frequency=1)
# steppableRegistry.registerSteppable(steppableInstance)
#
# CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)
#
#