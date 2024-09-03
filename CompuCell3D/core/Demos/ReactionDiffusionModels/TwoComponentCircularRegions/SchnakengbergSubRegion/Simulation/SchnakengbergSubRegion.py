from cc3d import CompuCellSetup

from SchnakengbergSubRegionSteppables import SchnakengbergSubRegionSteppable

CompuCellSetup.register_steppable(steppable=SchnakengbergSubRegionSteppable(frequency=1))

CompuCellSetup.run()
