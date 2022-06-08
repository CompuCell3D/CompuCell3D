
from cc3d import CompuCellSetup
        

from CellGDerivedPropertiesSteppables import CellGDerivedPropertiesSteppable

CompuCellSetup.register_steppable(steppable=CellGDerivedPropertiesSteppable(frequency=1))


CompuCellSetup.run()
