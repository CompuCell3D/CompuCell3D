from cc3d import CompuCellSetup
from .elongationTest_steering_steppables import LengthConstraintSteering

CompuCellSetup.register_steppable(steppable=LengthConstraintSteering(frequency=100))

CompuCellSetup.run()