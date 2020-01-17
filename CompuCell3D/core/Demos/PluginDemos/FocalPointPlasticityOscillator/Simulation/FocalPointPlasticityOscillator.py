
from cc3d import CompuCellSetup
        

from FocalPointPlasticityOscillatorSteppables import FocalPointPlasticityOscillatorSteppable

CompuCellSetup.register_steppable(steppable=FocalPointPlasticityOscillatorSteppable(frequency=1))


CompuCellSetup.run()
