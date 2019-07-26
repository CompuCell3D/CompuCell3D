from cc3d import CompuCellSetup
from .PixelTrackerExampleSteppables import PixelTrackerSteppable

CompuCellSetup.register_steppable(steppable=PixelTrackerSteppable(frequency=10))

CompuCellSetup.run()

