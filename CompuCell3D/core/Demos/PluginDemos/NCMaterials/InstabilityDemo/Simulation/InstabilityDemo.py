from cc3d import CompuCellSetup

from InstabilityDemoSteppables import InstabilityDemoSteppable
CompuCellSetup.register_steppable(steppable=InstabilityDemoSteppable(frequency=1))

CompuCellSetup.run()