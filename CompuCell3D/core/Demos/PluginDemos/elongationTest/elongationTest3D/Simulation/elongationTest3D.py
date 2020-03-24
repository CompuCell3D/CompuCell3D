from cc3d import CompuCellSetup

from elongationTest3DSteppables import elongationTest3DSteppable

CompuCellSetup.register_steppable(steppable=elongationTest3DSteppable(frequency=1))

CompuCellSetup.run()
