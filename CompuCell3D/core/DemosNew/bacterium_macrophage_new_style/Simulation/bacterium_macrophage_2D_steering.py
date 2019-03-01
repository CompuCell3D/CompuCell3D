import cc3d.CompuCellSetup as CompuCellSetup
from .bacterium_macrophage_2D_steppables import InventoryCheckSteppable

CompuCellSetup.register_steppable(steppable=InventoryCheckSteppable(frequency=1))

CompuCellSetup.run()

