
from cc3d import CompuCellSetup
        

from FocalPointPlasticityLinksSteppables import FocalPointPlasticityLinksSteppable

CompuCellSetup.register_steppable(steppable=FocalPointPlasticityLinksSteppable(frequency=1))


CompuCellSetup.run()
