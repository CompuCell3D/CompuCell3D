from cc3d import CompuCellSetup
from .steppableBasedMitosisSteppables import VolumeParamSteppable
from .steppableBasedMitosisSteppables import MitosisSteppable

CompuCellSetup.register_steppable(steppable=VolumeParamSteppable(frequency=10))
CompuCellSetup.register_steppable(steppable=MitosisSteppable(frequency=10))

CompuCellSetup.run()
