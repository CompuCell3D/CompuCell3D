from cc3d import CompuCellSetup
from .FieldSecretionSteppables import FieldSecretionSteppable

CompuCellSetup.register_steppable(steppable=FieldSecretionSteppable(frequency=10))

CompuCellSetup.run()

