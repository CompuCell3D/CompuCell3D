import cc3d.CompuCellSetup as CompuCellSetup
from .FocalPointPlasticityAnchorsSteppables import FocalPointPlasticityAnchorSteppable

CompuCellSetup.register_steppable(steppable=FocalPointPlasticityAnchorSteppable(frequency=10))

CompuCellSetup.run()

