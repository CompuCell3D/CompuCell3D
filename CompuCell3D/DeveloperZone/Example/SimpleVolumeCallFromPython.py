
import sys
from os import environ
import string
sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup

sim,simthread = CompuCellSetup.getCoreSimulationObjects()

import CompuCellExtraModules
import CompuCell

# plugin=sim.pluginManager.get("NeighborTracker")
plugin=CompuCell.getPlugin("SimpleVolume")
simpleVolume=CompuCellExtraModules.reinterpretSimpleVolumePlugin(plugin)
print "simpleVolume.toString()=",simpleVolume.toString()


