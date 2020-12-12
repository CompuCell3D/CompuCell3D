from cc3d import CompuCellSetup
from .chemotaxis_by_cell_id_steppables import ChemotaxisSteering

CompuCellSetup.register_steppable(steppable=ChemotaxisSteering(frequency=100))

CompuCellSetup.run()

