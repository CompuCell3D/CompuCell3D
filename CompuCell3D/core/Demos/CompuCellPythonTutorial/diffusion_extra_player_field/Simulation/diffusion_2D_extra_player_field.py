from cc3d import CompuCellSetup
from .diffusion_2D_steppables_player import ExtraFieldVisualizationSteppable

CompuCellSetup.register_steppable(steppable=ExtraFieldVisualizationSteppable(frequency=10))

CompuCellSetup.run()
