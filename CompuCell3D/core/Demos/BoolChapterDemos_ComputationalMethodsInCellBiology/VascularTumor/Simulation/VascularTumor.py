import sys
from os import environ
import string
sys.path.append(environ["PYTHON_MODULE_PATH"])



import CompuCellSetup
sim,simthread = CompuCellSetup.getCoreSimulationObjects()
CompuCellSetup.initializeSimulationObjects(sim,simthread)
import CompuCell

#Create extra player fields here or add attributes


from PySteppables import SteppableRegistry
steppableRegistry=SteppableRegistry()


from VascularTumorSteppables import MitosisSteppable
mitosisSteppable=MitosisSteppable(sim,1)
steppableRegistry.registerSteppable(mitosisSteppable)

from VascularTumorSteppables import VolumeParamSteppable
                                         #sim,frequency,areaThresh,nutrientThresh,necroticThresh
volumeParamSteppable=VolumeParamSteppable(sim,1)
steppableRegistry.registerSteppable(volumeParamSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

