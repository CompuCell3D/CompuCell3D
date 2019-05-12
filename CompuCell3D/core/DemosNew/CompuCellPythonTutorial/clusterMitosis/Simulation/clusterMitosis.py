from cc3d import CompuCellSetup
from .clusterMitosisSteppables import VolumeParamSteppable
from .clusterMitosisSteppables import MitosisSteppableClusters

CompuCellSetup.register_steppable(steppable=VolumeParamSteppable(frequency=10))
CompuCellSetup.register_steppable(steppable=MitosisSteppableClusters(frequency=10))

CompuCellSetup.run()


