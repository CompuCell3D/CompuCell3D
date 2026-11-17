from cc3d import CompuCellSetup
from .RuntimeDiscoverySteppables import RuntimeDiscoverySteppable


CompuCellSetup.register_compiled_steppable(
    library_path='Native/RuntimeDiscoveryABISteppable', frequency=1
)
CompuCellSetup.register_steppable(
    steppable=RuntimeDiscoverySteppable(frequency=1)
)
CompuCellSetup.run()
