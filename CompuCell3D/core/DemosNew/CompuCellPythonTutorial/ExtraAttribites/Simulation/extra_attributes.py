import cc3d.CompuCellSetup as CompuCellSetup

from .extra_attributes_steppables import ExtraAttributeCellsort
from .extra_attributes_steppables import TypeSwitcherSteppable


CompuCellSetup.register_steppable(steppable=ExtraAttributeCellsort(frequency=1))
CompuCellSetup.register_steppable(steppable=TypeSwitcherSteppable(frequency=1))

CompuCellSetup.run()





