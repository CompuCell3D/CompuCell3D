from cc3d import CompuCellSetup
from .CellMotilitySteppables import CellMotilitySteppable

CompuCellSetup.register_steppable(steppable=CellMotilitySteppable(frequency=10))

CompuCellSetup.run()

