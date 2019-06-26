from cc3d import CompuCellSetup
from .ChemotaxisCompartmentSteppables import ChemotaxisCompartmentSteppable

CompuCellSetup.register_steppable(steppable=ChemotaxisCompartmentSteppable(frequency=1))

CompuCellSetup.run()

