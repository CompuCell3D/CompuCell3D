from cc3d import CompuCellSetup

CompuCellSetup.register_compiled_steppable(
    library_path='Native/MitosisCompiledSteppable', frequency=10)
CompuCellSetup.run()
