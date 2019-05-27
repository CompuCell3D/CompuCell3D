from cc3d import CompuCellSetup
from .FluctuationAmplitudeSteppables import FluctuationAmplitude

CompuCellSetup.register_steppable(steppable=FluctuationAmplitude(frequency=100))

CompuCellSetup.run()

