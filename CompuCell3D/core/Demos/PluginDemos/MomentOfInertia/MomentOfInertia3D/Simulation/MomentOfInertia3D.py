from cc3d import CompuCellSetup
from .MomentOfInertia3DSteppables import MomentOfInertiaPrinter3D

CompuCellSetup.register_steppable(steppable=MomentOfInertiaPrinter3D(frequency=10))

CompuCellSetup.run()
