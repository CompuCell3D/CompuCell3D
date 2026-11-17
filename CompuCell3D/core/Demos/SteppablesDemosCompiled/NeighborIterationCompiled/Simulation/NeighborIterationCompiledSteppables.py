from cc3d.core.PySteppables import SteppableBasePy


class NeighborIterationVerifierSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        super().__init__(frequency=frequency)
        self._reported_missing_tracker = False

    def start(self):
        if self.neighbor_tracker_plugin:
            print('Python neighbor verifier start: NeighborTracker loaded')
        else:
            print('Python neighbor verifier start: NeighborTracker not loaded')

    def step(self, mcs):
        for i, cell in enumerate(self.cell_list):
            if i>=2:
                break

            if not self.neighbor_tracker_plugin:
                if not self._reported_missing_tracker:
                    print(f'Python neighbor verifier step mcs={mcs}: cannot inspect neighbors without NeighborTracker')
                    self._reported_missing_tracker = True
                return

            neighbor_count = 0
            medium_contacts = 0
            total_area = 0
            for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                neighbor_count += 1
                total_area += common_surface_area
                if neighbor is None:
                    medium_contacts += 1

            print(
                f'Python neighbor verifier step mcs={mcs} '
                f'cell_id={cell.id} neighbors={neighbor_count} '
                f'medium_contacts={medium_contacts} sharedSurface={total_area}'
            )


