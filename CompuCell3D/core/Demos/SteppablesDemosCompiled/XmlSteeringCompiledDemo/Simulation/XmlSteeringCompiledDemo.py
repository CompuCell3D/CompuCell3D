from cc3d import CompuCellSetup


CompuCellSetup.register_compiled_steppable(
    library_path="Native/XmlSteeringCompiledSteppable", frequency=1
)
CompuCellSetup.run()
