from cc3d import CompuCellSetup
from .SecretionCompiledDemoSteppables import SecretionVerifierSteppable


CompuCellSetup.register_compiled_steppable(
    library_path='Native/SecretionCompiledSteppable',
    frequency=1,
)
CompuCellSetup.register_steppable(steppable=SecretionVerifierSteppable(frequency=50))

CompuCellSetup.run()
