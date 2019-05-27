import cc3d.CompuCellSetup as CompuCellSetup
from .SteeringVolumeFlexSteppables import SteeringVolumeFlexSteppable

CompuCellSetup.register_steppable(steppable=SteeringVolumeFlexSteppable(frequency=1))

CompuCellSetup.run()