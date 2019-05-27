from cc3d import CompuCellSetup
from .SecretionSteadyStateSteppables import DiffusionFieldSteppable

CompuCellSetup.register_steppable(steppable=DiffusionFieldSteppable(frequency=1))

CompuCellSetup.run()

