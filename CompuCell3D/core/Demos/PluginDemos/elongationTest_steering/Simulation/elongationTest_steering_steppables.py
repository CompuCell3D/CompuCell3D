from cc3d.core.PySteppables import *


class LengthConstraintSteering(SteppableBasePy):
    def __init__(self, frequency=100):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        if mcs > 100 and not mcs % 100:
            lep_body1_elem = self.get_xml_element('lep_body1_elem')
            lep_body1_elem.TargetLength = float(lep_body1_elem.TargetLength) + 0.5

        if mcs > 3000:
            connect_elem = self.get_xml_element('connect_elem')
            connect_elem.cdata = 0
