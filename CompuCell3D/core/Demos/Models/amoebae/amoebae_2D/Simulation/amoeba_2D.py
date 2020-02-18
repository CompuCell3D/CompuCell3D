from cc3d import CompuCellSetup

from amoebae_2DSteppables import amoebae_2DSteppable

CompuCellSetup.register_steppable(steppable=amoebae_2DSteppable(frequency=1))

CompuCellSetup.run()
