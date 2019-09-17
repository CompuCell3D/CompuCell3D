import cc3d.CompuCellSetup as CompuCellSetup
from .cell_attribute_tracker_steppables import CellAttributeTracker

CompuCellSetup.register_steppable(steppable=CellAttributeTracker(frequency=10))

CompuCellSetup.run()





