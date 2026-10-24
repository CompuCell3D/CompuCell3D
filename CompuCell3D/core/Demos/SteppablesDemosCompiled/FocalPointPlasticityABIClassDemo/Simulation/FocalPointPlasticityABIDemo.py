from cc3d import CompuCellSetup

CompuCellSetup.register_compiled_steppable(
    library_path='Native/FocalPointPlasticityABIClassSteppable',
    frequency=1,
)
CompuCellSetup.run()

