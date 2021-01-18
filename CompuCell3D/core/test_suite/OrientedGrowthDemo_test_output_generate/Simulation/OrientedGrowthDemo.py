from cc3d import CompuCellSetup
from .OrientedGrowthDemoSteppables import OrientedConstraintSteppable

CompuCellSetup.register_steppable(steppable=OrientedConstraintSteppable(frequency=1))

CompuCellSetup.run()