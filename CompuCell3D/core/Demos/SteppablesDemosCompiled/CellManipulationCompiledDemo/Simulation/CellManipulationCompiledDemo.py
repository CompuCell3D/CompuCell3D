from cc3d import CompuCellSetup


CompuCellSetup.register_compiled_steppable(
    library_path='Native/CellManipulationCompiledSteppable',
    frequency=1
)

CompuCellSetup.run()
