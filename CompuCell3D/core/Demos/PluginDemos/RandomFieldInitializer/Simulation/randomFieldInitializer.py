from cc3d import CompuCellSetup

from randomFieldInitializerSteppables import randomFieldInitializerSteppable

CompuCellSetup.register_steppable(steppable=randomFieldInitializerSteppable(frequency=1))

CompuCellSetup.run()
