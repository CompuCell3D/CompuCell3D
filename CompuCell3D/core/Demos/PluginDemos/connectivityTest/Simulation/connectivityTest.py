from cc3d import CompuCellSetup

from connectivityTestSteppables import connectivityTestSteppable

CompuCellSetup.register_steppable(steppable=connectivityTestSteppable(frequency=1))

CompuCellSetup.run()
