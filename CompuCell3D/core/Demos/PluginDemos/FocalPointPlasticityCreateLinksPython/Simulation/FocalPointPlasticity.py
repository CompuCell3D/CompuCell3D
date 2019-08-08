import cc3d.CompuCellSetup as CompuCellSetup

from .FocalPointPlasticitySteppables import FocalPointPlasticityParams

CompuCellSetup.register_steppable(steppable=FocalPointPlasticityParams(frequency=10))


CompuCellSetup.run()

