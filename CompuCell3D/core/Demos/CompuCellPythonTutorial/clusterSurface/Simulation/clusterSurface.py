from cc3d import CompuCellSetup
from .clusterSurfaceSteppables import VolumeParamSteppable
from .clusterSurfaceSteppables import MitosisSteppableClusters

CompuCellSetup.register_steppable(steppable=VolumeParamSteppable(frequency=10))
CompuCellSetup.register_steppable(steppable=MitosisSteppableClusters(frequency=10))

CompuCellSetup.run()

