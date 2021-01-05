"""Tests all xml importing in cc3d.core.PyCoreSpecs"""

from os import listdir, walk
from os.path import abspath, dirname, join, splitext

from cc3d.core.PyCoreSpecs import from_file


tests_dir = join(dirname(dirname(dirname(abspath(__file__)))), "CompuCell3D", "core", "Demos")
"""Directory containing .cc3d files against which to test xml importing"""

patterns_ignore = [
    r"ArticlesPublishedDemos\VascularizedTumor3D_PlosOne2009",  # Redundant declaration in CellType Plugin
    r"BookChapterDemos_ComputationalMethodsInCellBiology\Angiogenesis",  # No XML
    r"BookChapterDemos_ComputationalMethodsInCellBiology\DeltaNotch",  # No XML
    r"BookChapterDemos_ComputationalMethodsInCellBiology\VascularTumor",  # Redundant declaration in Chemotaxis Plugin
    r"BookChapterDemos_ComputationalMethodsInCellBiology\VascularTumor_legacy_implementation",  # Redundant declaration in Chemotaxis Plugin
    r"CompuCellPythonTutorial\CellInitializer",  # Missing XML
    r"SimulationSettings\ParallelCC3DExamples\diffusion-3D",  # No XML
    r"SteppableDemos\DiffusionSolverFE\diffusion_3D_scale_wall"  # Incorrect id declaration in CellType Plugin
]
"""List of patterns to ignore during testing"""
patterns_ignore = [join(tests_dir, x) for x in patterns_ignore]
for f in listdir(join(tests_dir, "CompuCellPythonTutorial", "PythonOnlySimulationsExamples")):
    patterns_ignore.append(join(tests_dir, "CompuCellPythonTutorial", "PythonOnlySimulationsExamples", f))  # No XML


def main():
    """
    Test PyCoreSpecs module xml import against all demos

    :return: None
    """
    for dirpath, dirnames, filenames in walk(tests_dir):
        for fn in filenames:
            basename, ext = splitext(fn)
            if ext == ".xml":
                cc3d_filename = join(dirpath, fn)

                if dirname(dirname(cc3d_filename)) in patterns_ignore:
                    continue

            elif ext == ".cc3d":
                cc3d_filename = join(dirpath, fn)

                if dirname(cc3d_filename) in patterns_ignore:
                    continue

            else:
                continue

            print("Testing::", cc3d_filename)
            from_file(cc3d_filename)


if __name__ == "__main__":
    main()
