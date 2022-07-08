from cc3d import CompuCellSetup
from .message_window_steppables import MessageWindowSteppable

CompuCellSetup.register_steppable(steppable=MessageWindowSteppable(frequency=10))

CompuCellSetup.run()

