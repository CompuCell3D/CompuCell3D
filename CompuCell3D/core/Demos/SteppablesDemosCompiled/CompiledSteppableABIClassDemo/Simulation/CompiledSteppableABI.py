from cc3d import CompuCellSetup
from .CompiledSteppableABISteppables import TargetVolumeVerifierSteppable


CompuCellSetup.register_compiled_steppable(
    library_path='Native/GrowthClassSteppable',
    frequency=1
)
CompuCellSetup.register_steppable(steppable=TargetVolumeVerifierSteppable(frequency=10))

CompuCellSetup.run()

