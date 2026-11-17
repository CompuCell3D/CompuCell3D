from cc3d.core.PySteppables import SteppableBasePy


class SecretionVerifierSteppable(SteppableBasePy):
    """Checks the observable effects of the compiled Secretion steppable."""

    def __init__(self, frequency=50):
        super().__init__(frequency)

    def _first_bacterium(self):
        return next((cell for cell in self.cell_list if cell.type == self.BACTERIUM), None)

    def start(self):
        pass

    def step(self, mcs):
        cell = self._first_bacterium()
        secretor = self.get_field_secretor("ATTR")
        seen = secretor.amountSeenByCell(cell) if cell is not None else 0.0
        integral = secretor.totalFieldIntegral()
        if seen <= 0.0 or integral <= 0.0:
            raise RuntimeError(
                f"Compiled Secretion step check failed at MCS {mcs}: "
                f"amount_seen={seen}, integral={integral}"
            )
        print(f"Secretion verifier: MCS {mcs} passed (amount_seen={seen}, integral={integral})")
