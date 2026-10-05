from cc3d import CompuCellSetup
from .GeometryComparisonSteppables import GeometryComparisonSteppable

CompuCellSetup.register_compiled_steppable(
    library_path='Native/GeometryABISteppable',
    frequency=1,
)
CompuCellSetup.register_steppable(steppable=GeometryComparisonSteppable(frequency=1))

CompuCellSetup.run()
