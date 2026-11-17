from cc3d import CompuCellSetup
from .CompiledSteppableABIFieldsSteppables import FieldDrivenGrowthVerifierSteppable


CompuCellSetup.register_compiled_steppable(
    library_path='Native/FieldDrivenGrowthClassSteppable',
    frequency=1
)
CompuCellSetup.register_steppable(steppable=FieldDrivenGrowthVerifierSteppable(frequency=1))

CompuCellSetup.run()

