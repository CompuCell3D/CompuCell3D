from cc3d import CompuCellSetup

from elongationTestSteppables import elongationTestSteppable

CompuCellSetup.register_steppable(steppable=elongationTestSteppable(frequency=1))

CompuCellSetup.run()
