from cc3d.core.PySteppables import *


class SteeringVolumeFlexSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        pass

    def step(self, mcs):
        print('step=', mcs)

        if mcs == 1:

            temp_elem = self.get_xml_element('temp')

            temp_elem.cdata = 100

        if mcs == 10:
            self.get_xml_element('temp').cdata = 1

        if mcs == 20:
            vol_cond_elem = self.get_xml_element('vol_cond')
            vol_cond_elem.TargetVolume = 50


