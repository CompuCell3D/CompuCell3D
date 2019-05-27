from cc3d import CompuCellSetup
from .CellDistanceSteppables import CellDistanceSteppable

CompuCellSetup.register_steppable(steppable=CellDistanceSteppable(frequency=1))

CompuCellSetup.run()

