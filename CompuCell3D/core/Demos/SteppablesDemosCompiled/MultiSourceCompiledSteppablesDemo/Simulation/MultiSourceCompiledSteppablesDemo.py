from cc3d import CompuCellSetup
from .MultiSourceCompiledSteppablesSteppables import (
    SharedNumpyFieldsSteppable,
    TargetVolumeVerifierSteppable,
)


CompuCellSetup.register_compiled_steppable(
    library_path="Native_GrowthSteppable/GrowthClassSteppable",
    frequency=1,
)
CompuCellSetup.register_compiled_steppable(
    library_path="Native_SharedNumpyFieldsSteppable/SharedNumpyFieldsCompiledSteppable",
    frequency=1,
)
CompuCellSetup.register_steppable(
    steppable=SharedNumpyFieldsSteppable(frequency=1)
)
CompuCellSetup.register_steppable(
    steppable=TargetVolumeVerifierSteppable(frequency=10)
)
CompuCellSetup.run()
