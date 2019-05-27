from cc3d.core.PySteppables import *


class ElasticitySteering(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        if (100 < mcs < 2000) and not mcs % 100:
            tgt_len_elem = self.get_xml_element('tgt_len_elem')
            tgt_len_elem.cdata = float(tgt_len_elem.cdata) + 1

