from cc3d import CompuCellSetup
from .CellFieldOperationsSteppables import CellFieldOperationsSteppable


CompuCellSetup.register_compiled_steppable(
    library_path='Native/CellFieldOperationsSteppableCompiled',
    frequency=1
)
CompuCellSetup.register_steppable(steppable=CellFieldOperationsSteppable(frequency=1))

CompuCellSetup.run()
