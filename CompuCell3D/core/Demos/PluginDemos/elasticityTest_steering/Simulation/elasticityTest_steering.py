from cc3d import CompuCellSetup
from .elasticityTest_steering_steppables import ElasticitySteering

CompuCellSetup.register_steppable(steppable=ElasticitySteering(frequency=100))

CompuCellSetup.run()
