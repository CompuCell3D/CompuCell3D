import unittest
from pathlib import Path


class TestCompiledSteppableABI(unittest.TestCase):
    def setUp(self):
        from cc3d import CompuCellSetup
        CompuCellSetup.resetGlobals()

    def test_register_compiled_steppable_wrapper(self):
        from cc3d import CompuCellSetup
        from cc3d.core.CompiledSteppable import CompiledSteppable

        steppable = CompuCellSetup.register_compiled_steppable(
            library_path='missing_compiled_steppable.so',
            frequency=7,
            name='MissingSteppable'
        )

        self.assertIsInstance(steppable, CompiledSteppable)
        self.assertEqual(steppable.frequency, 7)
        self.assertEqual(steppable.name, 'MissingSteppable')
        self.assertTrue(steppable.library_path.endswith('missing_compiled_steppable.so'))

    def tearDown(self):
        from cc3d import CompuCellSetup
        CompuCellSetup.resetGlobals()

    def test_native_loader_reports_missing_module(self):
        from cc3d.cpp import CompuCell

        missing_lib = str(Path('missing_compiled_steppable.so').resolve())
        compiled = CompuCell.CompiledSteppable(missing_lib)

        with self.assertRaises(RuntimeError):
            compiled.load()
