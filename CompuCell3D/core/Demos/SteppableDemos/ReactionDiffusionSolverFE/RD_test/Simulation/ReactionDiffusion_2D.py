from cc3d import CompuCellSetup
from .RD_steppables import FieldSteppable

CompuCellSetup.register_steppable(steppable=FieldSteppable(frequency=1))

CompuCellSetup.run()
