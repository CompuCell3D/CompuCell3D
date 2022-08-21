from weakref import ref


class BasicSimulationData:
    def __init__(self):
        self._sim = None
        self._field_dim = None
        self.cell_types_used = None
        self.numberOfSteps = 0
        self.current_step = 0

    @property
    def sim(self):
        if self._sim is None:
            from cc3d.CompuCellSetup import persistent_globals
            if persistent_globals.simulator is None:
                return None
            self._sim = ref(persistent_globals.simulator)
        return self._sim()

    @sim.setter
    def sim(self, _sim):
        self._sim = ref(_sim) if _sim is not None else None

    @property
    def fieldDim(self):
        if self._field_dim is None:
            try:
                self._field_dim = self.sim.getPotts().getCellFieldG().getDim()
            except AttributeError:
                return None
        return self._field_dim

    @fieldDim.setter
    def fieldDim(self, _field_dim):
        self._field_dim = _field_dim
