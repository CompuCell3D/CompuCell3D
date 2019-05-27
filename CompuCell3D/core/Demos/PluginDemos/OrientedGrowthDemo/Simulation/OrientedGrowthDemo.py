from cc3d import CompuCellSetup
from .OrientedGrowthDemoSteppables import GrowthSteppable
from .OrientedGrowthDemoSteppables import OrientedConstraintSteppable

CompuCellSetup.register_steppable(steppable=GrowthSteppable(frequency=10))
CompuCellSetup.register_steppable(steppable=OrientedConstraintSteppable(frequency=1))

CompuCellSetup.run()