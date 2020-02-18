from cc3d import CompuCellSetup


from amoebae_2D_secretionSteppables import amoebae_2D_secretionSteppable

CompuCellSetup.register_steppable(steppable=amoebae_2D_secretionSteppable(frequency=1))


CompuCellSetup.run()
