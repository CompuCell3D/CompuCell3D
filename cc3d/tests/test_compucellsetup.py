# cc3d/tests/test_compucellsetup.py
from cc3d.CompuCellSetup import getCore

def test_get_core_returns_object():
    core = getCore()
    assert core is not None

