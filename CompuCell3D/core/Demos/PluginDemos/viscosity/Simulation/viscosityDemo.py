from cc3d import CompuCellSetup

from viscosityDemoSteppables import viscosityDemoSteppable

CompuCellSetup.register_steppable(steppable=viscosityDemoSteppable(frequency=1))

CompuCellSetup.run()
