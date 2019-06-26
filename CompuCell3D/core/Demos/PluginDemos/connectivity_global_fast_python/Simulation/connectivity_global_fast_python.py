from cc3d import CompuCellSetup
from .connectivity_global_fast_python_steppables import ConnectivitySteppable

CompuCellSetup.register_steppable(steppable=ConnectivitySteppable(frequency=1))

CompuCellSetup.run()
