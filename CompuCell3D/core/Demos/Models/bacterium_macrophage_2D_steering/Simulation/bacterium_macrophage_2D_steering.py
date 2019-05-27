from cc3d import CompuCellSetup
from .bacterium_macrophage_2D_steering_steppables import ChemotaxisSteering

CompuCellSetup.register_steppable(steppable=ChemotaxisSteering(frequency=100))

CompuCellSetup.run()