from cc3d import CompuCellSetup

from .amoeba_2DSteppables import amoeba_2DSteppable

CompuCellSetup.register_steppable(steppable=amoeba_2DSteppable(frequency=1))

CompuCellSetup.run()
