from cc3d import CompuCellSetup

from bacterium_macrophage_2DSteppables import bacterium_macrophage_2DSteppable

CompuCellSetup.register_steppable(steppable=bacterium_macrophage_2DSteppable(frequency=1))


CompuCellSetup.run()
