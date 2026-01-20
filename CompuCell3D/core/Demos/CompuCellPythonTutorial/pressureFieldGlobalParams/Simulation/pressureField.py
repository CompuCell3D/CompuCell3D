from cc3d import CompuCellSetup
from .pressureFieldSteppables import TargetVolumeDrosoSteppable
from .pressureFieldSteppables import CellKiller
from .pressureFieldSteppables import PressureFieldVisualizationSteppable

CompuCellSetup.register_steppable(steppable=TargetVolumeDrosoSteppable(frequency=1))
CompuCellSetup.register_steppable(steppable=CellKiller(frequency=1))
CompuCellSetup.register_steppable(steppable=PressureFieldVisualizationSteppable(frequency=10))


CompuCellSetup.run()


