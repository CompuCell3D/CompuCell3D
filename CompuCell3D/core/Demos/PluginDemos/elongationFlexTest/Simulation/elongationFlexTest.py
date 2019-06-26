from cc3d import CompuCellSetup
from .elongationFlexSteppables import ElongationFlexSteppable

CompuCellSetup.register_steppable(steppable=ElongationFlexSteppable(frequency=50))

CompuCellSetup.run()