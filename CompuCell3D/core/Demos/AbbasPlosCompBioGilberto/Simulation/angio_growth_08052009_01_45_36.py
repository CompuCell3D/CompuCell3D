from .angio_growth_steppables_08052009_01_45_36 import VolumeParamSteppable
from .angio_growth_steppables_08052009_01_45_36 import MitosisSteppable
import cc3d.CompuCellSetup as CompuCellSetup

vol_steppable = VolumeParamSteppable(frequency=1)
vol_steppable.set_params(1, 5, 1)
CompuCellSetup.register_steppable(steppable=vol_steppable)

doublingVolumeDict = {1: 54, 2: 54, 4: 80, 6: 80}
mitosis_steppable = MitosisSteppable(frequency=1)

mitosis_steppable.set_params(doublingVolumeDict=doublingVolumeDict)
CompuCellSetup.register_steppable(steppable=mitosis_steppable)


CompuCellSetup.run()

# ####
# #### The simulation code is compatible with CompuCell3D ver 3.3.1
# ####
#
# import sys
# from os import environ
# import string
# sys.path.append(environ["PYTHON_MODULE_PATH"])
#
# #
#
# import CompuCellSetup
#
# sim,simthread = CompuCellSetup.getCoreSimulationObjects()
#
# #Create extra player fields here or add attributes
#
# pyAttributeAdder,listAdder=CompuCellSetup.attachListToCells(sim)
#
# CompuCellSetup.initializeSimulationObjects(sim,simthread)
#
# #
# ##################
# ##########	PLUGINS
# ##################
# #
#
# import CompuCell
#
# from angio_growth_plugins_08052009_01_45_36 import *
#
# changeWatcherRegistry=CompuCellSetup.getChangeWatcherRegistry(sim)
#
# stepperRegistry=CompuCellSetup.getStepperRegistry(sim)
#
#
# mitPy=MitosisPyPlugin(sim,changeWatcherRegistry,stepperRegistry)
#
# #### seting doubling volumes for normal, hypoxic, ActiveNeovascular, InactiveNeovascular
# doublingVolumeDict = {1:54,2:54,4:80,6:80}
# mitPy.setCellDoublingVolume(doublingVolumeDict)
#
# #
# ##################
# ##########	STEPPABLES
# ##################
# #
#
#
# from PySteppables import SteppableRegistry
# steppableRegistry=SteppableRegistry()
#
# from angio_growth_steppables_08052009_01_45_36 import *
#                                          #sim,frequency,areaThresh,nutrientThresh,necroticThresh
# volumeParamSteppable=VolumeParamSteppable(sim,1,1,5,1)
# steppableRegistry.registerSteppable(volumeParamSteppable)
#
#
# #
# ##################
# ##########	COMPUCELL3D LOOPS
# ##################
# #
#
# CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)
#
