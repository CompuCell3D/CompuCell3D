from .TumorVasc3DSteppables import VolumeParamSteppable
from .TumorVasc3DSteppables import MitosisSteppable
import cc3d.CompuCellSetup as CompuCellSetup

vol_steppable = VolumeParamSteppable(frequency=1)
vol_steppable.set_params(1, 5, 1)
CompuCellSetup.register_steppable(steppable=vol_steppable)

doublingVolumeDict = {1: 54, 2: 54, 4: 80, 6: 80}
mitosis_steppable = MitosisSteppable(frequency=1)

mitosis_steppable.set_params(doublingVolumeDict=doublingVolumeDict)
CompuCellSetup.register_steppable(steppable=mitosis_steppable)

CompuCellSetup.run()

