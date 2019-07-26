import sys
from os import environ
sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup

sim, simthread = CompuCellSetup.getCoreSimulationObjects()

CompuCellSetup.initializeSimulationObjects(sim, simthread)

# Add Python steppables here
steppableRegistry = CompuCellSetup.getSteppableRegistry()

from scipy_demo_steppables import ScipyDemoSteppable

scipy_demo_steppable = ScipyDemoSteppable(sim, 100)
steppableRegistry.registerSteppable(scipy_demo_steppable)

CompuCellSetup.mainLoop(sim, simthread, steppableRegistry)
