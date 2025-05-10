
from cc3d import CompuCellSetup
        

from oscillatorypatternCC3Dv4Steppables import oscillatorypatternCC3Dv4Steppable

CompuCellSetup.register_steppable(steppable=oscillatorypatternCC3Dv4Steppable(frequency=1))


CompuCellSetup.run()
