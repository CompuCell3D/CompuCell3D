from cc3d import CompuCellSetup
from .SteppableCommunicationSteppables import CommunicationSteppable
from .SteppableCommunicationSteppables import ExtraSteppable

CompuCellSetup.register_steppable(steppable=CommunicationSteppable(frequency=1))
CompuCellSetup.register_steppable(steppable=ExtraSteppable(frequency=1))

CompuCellSetup.run()