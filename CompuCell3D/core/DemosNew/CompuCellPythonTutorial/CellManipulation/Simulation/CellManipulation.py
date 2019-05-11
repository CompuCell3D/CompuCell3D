from cc3d import CompuCellSetup
from .CellManipulationSteppables import CellManipulationSteppable

CompuCellSetup.register_steppable(steppable=CellManipulationSteppable(frequency=10))

CompuCellSetup.run()


