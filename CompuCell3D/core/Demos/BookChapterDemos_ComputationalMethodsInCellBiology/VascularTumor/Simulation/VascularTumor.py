from cc3d import CompuCellSetup
from .VascularTumorSteppables import MitosisSteppable
from .VascularTumorSteppables import VolumeParamSteppable

CompuCellSetup.register_steppable(steppable=MitosisSteppable(frequency=1))
CompuCellSetup.register_steppable(steppable=VolumeParamSteppable(frequency=1))

CompuCellSetup.run()

