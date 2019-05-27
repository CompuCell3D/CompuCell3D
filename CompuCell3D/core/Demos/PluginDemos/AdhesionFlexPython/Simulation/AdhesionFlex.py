from cc3d import CompuCellSetup
from .AdhesionFlexSteppables import AdhesionMoleculesSteppables

CompuCellSetup.register_steppable(steppable=AdhesionMoleculesSteppables(frequency=10))

CompuCellSetup.run()

