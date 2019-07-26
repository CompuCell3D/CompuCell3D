import cc3d.CompuCellSetup as CompuCellSetup
from .DeltaNotchSteppables import DNVisualizationSteppable
from .DeltaNotchSteppables import DeltaNotchClass

CompuCellSetup.register_steppable(steppable=DeltaNotchClass(frequency=1))
CompuCellSetup.register_steppable(steppable=DNVisualizationSteppable(frequency=1))

CompuCellSetup.run()

