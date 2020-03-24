
from cc3d import CompuCellSetup
        

from CC3DJitTestSteppables import CC3DJitTestSteppable

CompuCellSetup.register_steppable(steppable=CC3DJitTestSteppable(frequency=1))


CompuCellSetup.run()
