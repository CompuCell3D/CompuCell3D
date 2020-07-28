from cc3d import CompuCellSetup
from .on_stop_demo_steppables import VolumeParamSteppable
from .on_stop_demo_steppables import MitosisSteppable

CompuCellSetup.register_steppable(steppable=VolumeParamSteppable(frequency=10))
CompuCellSetup.register_steppable(steppable=MitosisSteppable(frequency=10))

CompuCellSetup.run()
