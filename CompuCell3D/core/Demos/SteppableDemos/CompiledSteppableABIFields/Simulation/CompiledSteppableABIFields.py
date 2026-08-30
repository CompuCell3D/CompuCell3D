from cc3d import CompuCellSetup
from .CompiledSteppableABIFieldsSteppables import FieldDrivenGrowthVerifierSteppable


def compiled_library_path():
    from pathlib import Path
    import sys

    cc3d_project = Path(CompuCellSetup.cc3dSimulationDataHandler.cc3dSimulationData.path)
    native_dir = cc3d_project.parent / 'Native'
    suffix = {
        'darwin': '.dylib',
        'win32': '.dll'
    }.get(sys.platform, '.so')
    return native_dir / f'FieldDrivenGrowthSteppable{suffix}'


CompuCellSetup.register_compiled_steppable(
    library_path=str(compiled_library_path()),
    frequency=1
)
CompuCellSetup.register_steppable(steppable=FieldDrivenGrowthVerifierSteppable(frequency=10))

CompuCellSetup.run()
