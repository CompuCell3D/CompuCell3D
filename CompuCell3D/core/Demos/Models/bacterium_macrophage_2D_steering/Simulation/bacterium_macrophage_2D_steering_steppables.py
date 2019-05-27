from cc3d.core.PySteppables import *


class ChemotaxisSteering(SteppableBasePy):
    def __init__(self, frequency=100):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        if mcs > 100 and not mcs % 100:
            vol_cond_elem = self.get_xml_element('macro_chem')
            vol_cond_elem.Lambda = float(vol_cond_elem.Lambda) - 3

