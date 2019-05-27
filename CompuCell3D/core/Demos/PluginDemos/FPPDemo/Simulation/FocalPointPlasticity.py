from cc3d import CompuCellSetup
from .FocalPointPlasticitySteppables import FocalPointPlasticityParams

CompuCellSetup.register_steppable(steppable=FocalPointPlasticityParams(frequency=10))

CompuCellSetup.run()
