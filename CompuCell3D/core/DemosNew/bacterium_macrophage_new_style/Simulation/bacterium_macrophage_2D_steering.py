import sys
import os
print ('sys.path=',sys.path)

print('os.environ=',os.environ)

# print('.', .bacterium_macrophage_2D_steppables)

from .bacterium_macrophage_2D_steppables import InventoryCheckSteppable
import cc3d.CompuCellSetup as CompuCellSetup

CompuCellSetup.register_steppable(steppable=InventoryCheckSteppable(frequency=1))

CompuCellSetup.run()

