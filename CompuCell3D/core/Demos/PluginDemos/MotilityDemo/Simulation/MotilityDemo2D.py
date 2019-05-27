from cc3d import CompuCellSetup
from .MotilityDemo2DSteppables import MotilityDemo2DSteppable

CompuCellSetup.register_steppable(steppable=MotilityDemo2DSteppable(frequency=1))

CompuCellSetup.run()

