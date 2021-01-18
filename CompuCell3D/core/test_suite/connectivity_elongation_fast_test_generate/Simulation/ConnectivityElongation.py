from cc3d import CompuCellSetup
from .ConnectivityElongationSteppable import ConnectivityElongationSteppable

CompuCellSetup.register_steppable(steppable=ConnectivityElongationSteppable(frequency=50))

CompuCellSetup.run()
