from cc3d import CompuCellSetup
from .bacterium_macrophage_2D_secretion_steppables import SecretionSteppable

CompuCellSetup.register_steppable(steppable=SecretionSteppable(frequency=1))

CompuCellSetup.run()

