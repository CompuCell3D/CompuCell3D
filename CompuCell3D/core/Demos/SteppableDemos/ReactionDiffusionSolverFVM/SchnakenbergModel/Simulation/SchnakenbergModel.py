
from cc3d import CompuCellSetup
        

from SchnakenbergModelSteppables import SchnakenbergModelSteppable

CompuCellSetup.register_steppable(steppable=SchnakenbergModelSteppable(frequency=1))


CompuCellSetup.run()
