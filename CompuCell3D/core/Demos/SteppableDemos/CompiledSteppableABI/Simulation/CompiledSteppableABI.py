from cc3d import CompuCellSetup
from .CompiledSteppableABISteppables import TargetVolumeVerifierSteppable


def compiled_library_path():
    from pathlib import Path
    import sys

    cc3d_project = Path(CompuCellSetup.cc3dSimulationDataHandler.cc3dSimulationData.path)
    native_dir = cc3d_project.parent / 'Native'
    suffix = {
        'darwin': '.dylib',
        'win32': '.dll'
    }.get(sys.platform, '.so')
    return native_dir / f'GrowthSteppable{suffix}'


CompuCellSetup.register_compiled_steppable(
    library_path=str(compiled_library_path()),
    frequency=1
)
CompuCellSetup.register_steppable(steppable=TargetVolumeVerifierSteppable(frequency=10))

CompuCellSetup.run()
# from pathlib import Path
# import sys
#
# from cc3d import CompuCellSetup
# from .CompiledSteppableABISteppables import TargetVolumeVerifierSteppable
#
#
# def compiled_library_path():
#     cc3d_project = Path(CompuCellSetup.cc3dSimulationDataHandler.cc3dSimulationData.path)
#     native_dir = cc3d_project.parent / 'Native'
#     suffix = {
#         'darwin': '.dylib',
#         'win32': '.dll'
#     }.get(sys.platform, '.so')
#     return native_dir / f'GrowthSteppable{suffix}'
#
#
# CompuCellSetup.register_compiled_steppable(
#     library_path=str(compiled_library_path()),
#     frequency=1
# )
# CompuCellSetup.register_steppable(steppable=TargetVolumeVerifierSteppable(frequency=10))
#
# CompuCellSetup.run()
# # from cc3d import CompuCellSetup
# # from .CompiledSteppableABISteppables import TargetVolumeVerifierSteppable
# #
# #
# # def compiled_library_path():
# #     from pathlib import Path
# #     import sys
# #
# #     native_dir = Path(__file__).resolve().parents[1] / 'Native'
# #     suffix = {
# #         'darwin': '.dylib',
# #         'win32': '.dll'
# #     }.get(sys.platform, '.so')
# #     return native_dir / f'GrowthSteppable{suffix}'
# #
# #
# # CompuCellSetup.register_compiled_steppable(
# #     library_path=str(compiled_library_path()),
# #     frequency=1
# # )
# # CompuCellSetup.register_steppable(steppable=TargetVolumeVerifierSteppable(frequency=10))
# #
# # CompuCellSetup.run()
# # # from pathlib import Path
# # # import sys
# # #
# # # from cc3d import CompuCellSetup
# # # from .CompiledSteppableABISteppables import TargetVolumeVerifierSteppable
# # #
# # #
# # # def compiled_library_path() -> Path:
# # #     native_dir = Path(__file__).resolve().parents[1] / 'Native'
# # #     suffix = {
# # #         'darwin': '.dylib',
# # #         'win32': '.dll'
# # #     }.get(sys.platform, '.so')
# # #     return native_dir / f'GrowthSteppable{suffix}'
# # #
# # #
# # # CompuCellSetup.register_compiled_steppable(
# # #     library_path=str(compiled_library_path()),
# # #     frequency=1
# # # )
# # # CompuCellSetup.register_steppable(steppable=TargetVolumeVerifierSteppable(frequency=10))
# # #
# # # CompuCellSetup.run()
