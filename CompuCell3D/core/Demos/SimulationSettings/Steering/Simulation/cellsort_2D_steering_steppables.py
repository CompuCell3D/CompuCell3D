from cc3d.core.PySteppables import *


class ContactSteeringAndTemperature(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        temp_elem = self.get_xml_element('temp')

        temp_elem.cdata = float(temp_elem.cdata) + 1

        contact_c_nonc_elem = self.get_xml_element('contact_c_nonc')
        contact_c_nonc_elem.cdata = float(contact_c_nonc_elem.cdata) + 1
