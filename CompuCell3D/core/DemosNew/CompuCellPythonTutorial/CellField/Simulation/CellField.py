import cc3d.CompuCellSetup as CompuCellSetup
from .CellFieldSteppables import UniformInitializer

CompuCellSetup.register_steppable(steppable=UniformInitializer(frequency=1))

CompuCellSetup.run()

