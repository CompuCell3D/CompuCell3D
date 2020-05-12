
from cc3d import CompuCellSetup
        

from SecretorFieldInspectionSteppables import SecretorFieldInspectionSteppable

CompuCellSetup.register_steppable(steppable=SecretorFieldInspectionSteppable(frequency=10))


CompuCellSetup.run()
