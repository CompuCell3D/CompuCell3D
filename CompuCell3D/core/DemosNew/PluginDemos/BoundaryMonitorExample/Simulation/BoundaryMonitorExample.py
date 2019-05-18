from cc3d import CompuCellSetup
from .BoundaryMonitorExampleSteppables import BoundaryMonitorSteppable

CompuCellSetup.register_steppable(steppable=BoundaryMonitorSteppable(frequency=1))

CompuCellSetup.run()

