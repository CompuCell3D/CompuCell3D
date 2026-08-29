from cc3d.core.PySteppables import SteppableBasePy


class TargetVolumeVerifierSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        super().__init__(frequency=frequency)

    def start(self):
        print('Python verifier start')

    def step(self, mcs):
        for cell in self.cell_list_by_type(self.A):
            print(
                f'Python verifier step mcs={mcs} '
                f'cell_id={cell.id} volume={cell.volume} targetVolume={cell.targetVolume}'
            )
            break

    def finish(self):
        print('Python verifier finish')
