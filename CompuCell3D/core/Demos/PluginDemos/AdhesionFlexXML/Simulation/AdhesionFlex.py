from cc3d import CompuCellSetup

from AdhesionFlexSteppables import AdhesionFlexSteppable

CompuCellSetup.register_steppable(steppable=AdhesionFlexSteppable(frequency=1))

CompuCellSetup.run()
