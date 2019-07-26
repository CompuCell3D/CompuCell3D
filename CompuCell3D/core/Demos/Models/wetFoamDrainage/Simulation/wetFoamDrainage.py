from cc3d import CompuCellSetup
from .wetFoamDrainageSteppables import FlexCellInitializer

fci = FlexCellInitializer(frequency=1)
fci.add_cell_type_parameters(cell_type=1, count=80, target_volume=25, lambda_volume=10.0)
fci.add_cell_type_parameters(cell_type=2, count=0, target_volume=5, lambda_volume=2.0)
fci.set_fraction_of_water(0.25)

CompuCellSetup.register_steppable(steppable=fci)

CompuCellSetup.run()
