from cc3d import CompuCellSetup
from .NeighborIterationCompiledSteppables import NeighborIterationVerifierSteppable


CompuCellSetup.register_compiled_steppable(
    library_path='Native/NeighborIterationClassSteppable',
    frequency=10
)
CompuCellSetup.register_steppable(steppable=NeighborIterationVerifierSteppable(frequency=10))

CompuCellSetup.run()

