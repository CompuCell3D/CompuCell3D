from cc3d import CompuCellSetup
from .MomentOfInertiaSteppables import MomentOfInertiaPrinter

CompuCellSetup.register_steppable(steppable=MomentOfInertiaPrinter(frequency=10))

CompuCellSetup.run()
