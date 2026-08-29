from pathlib import Path

from cc3d.cpp import CompuCell
from cc3d.core.PySteppables import SteppableBasePy


class CompiledSteppable(SteppableBasePy):
    """
    Python steppable wrapper for a compiled kernel.h-based steppable module.
    """

    def __init__(self, library_path: str, frequency: int = 1,
                 entry_point: str = 'cc3d_get_steppable_v1', name: str = None):
        super().__init__(frequency=frequency)
        self.library_path = str(Path(library_path).expanduser().resolve())
        self.entry_point = entry_point
        self.name = name or Path(self.library_path).stem
        self._compiled = CompuCell.CompiledSteppable(self.library_path, self.entry_point)

    def core_init(self, reinitialize_cell_types=True):
        super().core_init(reinitialize_cell_types=reinitialize_cell_types)
        self._compiled.attachSimulator(self.simulator)

    def start(self):
        self._compiled.start()

    def step(self, mcs):
        self._compiled.step(mcs)

    def finish(self):
        self._compiled.finish()

    def on_stop(self):
        self._compiled.cleanup()

    def cleanup(self):
        self._compiled.cleanup()
