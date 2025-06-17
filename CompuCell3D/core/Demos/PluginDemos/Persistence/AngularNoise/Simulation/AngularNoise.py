
from cc3d import CompuCellSetup
        

from AngularNoiseSteppables import AngularNoiseSteppable

CompuCellSetup.register_steppable(steppable=AngularNoiseSteppable(frequency=1))


CompuCellSetup.run()
