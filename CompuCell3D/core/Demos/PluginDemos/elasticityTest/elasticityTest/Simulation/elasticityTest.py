from cc3d import CompuCellSetup

from elasticityTestSteppables import elasticityTestSteppable

CompuCellSetup.register_steppable(steppable=elasticityTestSteppable(frequency=1))

CompuCellSetup.run()
