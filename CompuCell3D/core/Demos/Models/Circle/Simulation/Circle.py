
from cc3d import CompuCellSetup
        

from CircleSteppables import CircleSteppable

CompuCellSetup.register_steppable(steppable=CircleSteppable(frequency=1))


CompuCellSetup.run()
