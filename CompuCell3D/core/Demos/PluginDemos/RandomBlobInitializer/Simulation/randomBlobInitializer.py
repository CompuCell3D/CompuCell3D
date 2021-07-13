from cc3d import CompuCellSetup

from randomBlobInitializerSteppables import randomBlobInitializerSteppable

CompuCellSetup.register_steppable(steppable=randomBlobInitializerSteppable(frequency=1))

CompuCellSetup.run()
