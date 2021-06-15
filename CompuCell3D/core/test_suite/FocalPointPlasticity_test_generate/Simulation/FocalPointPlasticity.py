import cc3d.CompuCellSetup as CompuCellSetup

from FocalPointPlasticitySteppables import FocalPointPlasticitySteppable

CompuCellSetup.register_steppable(steppable=FocalPointPlasticitySteppable(frequency=1))

CompuCellSetup.run()

