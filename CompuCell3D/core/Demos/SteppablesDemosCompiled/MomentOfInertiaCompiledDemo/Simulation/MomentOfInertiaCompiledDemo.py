from cc3d import CompuCellSetup


CompuCellSetup.register_compiled_steppable(
    library_path='Native/MomentOfInertiaCompiledSteppable',
    frequency=10
)

CompuCellSetup.run()
