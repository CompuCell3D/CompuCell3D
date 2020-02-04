from cc3d import CompuCellSetup
from .screenshots_steppables import ScreenshotSteppable

CompuCellSetup.register_steppable(steppable=ScreenshotSteppable(frequency=1))

CompuCellSetup.run()
