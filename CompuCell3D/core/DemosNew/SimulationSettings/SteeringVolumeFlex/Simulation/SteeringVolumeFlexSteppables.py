from cc3d.core.PySteppables import *


class SteeringVolumeFlexSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        """
        Here is an example of how to modify values stored in the CC3D XML:
        1. First we add "id" tag to all XML elements we want to modify. Make sure the ids are unique
        For example we have for Temperature element
            <Temperature id="temp">10.0</Temperature>

        and for VolumeEnergyParameters
        <VolumeEnergyParameters id="vol_cond" CellType="Condensing" LambdaVolume="2.0" TargetVolume="25"/>


        2. Fetch element object in Python
        For example:

            temp_elem = self.get_xml_element('temp')

        or
            vol_cond_elem = self.get_xml_element('vol_cond')

        3. Change the value of the element (cdata) or attribute of the element
            a) changing value of the element .
            The value is defined as the string quantity that sits between > and <
            For example in <Temperature id="temp">10.0</Temperature> 10.0 is a value
            We change it using the following syntax

            temp_elem.cdata = float(temp_elem.cdata) + 10

            Notice that temp_elem.cdata is  a string - in general anything in XMl is a string so we first convert it to
            float and then assign back to cdata member

            b) If we want to change attribute we use syntax as below
            vol_cond_elem.TargetVolume = 2.0 * float(vol_cond_elem.TargetVolume)

            where we use name of the attribute as it appears in the XML

            In our example ,the element
            <VolumeEnergyParameters id="vol_cond" CellType="Condensing" LambdaVolume="2.0" TargetVolume="25"/>
            has 4 attributes id, CellType, LambdaVolume and TargetVolume. you should not change id attribute
            and also do not change CellType element. In this example changin g of lambda volume and target volume
            makes sense


        """

        temp_elem = self.get_xml_element('temp')
        temp_elem.cdata = float(temp_elem.cdata) + 10

        if mcs == 10:
            temp_elem = self.get_xml_element('temp')
            temp_elem.cdata = float(temp_elem.cdata) + 10

            vol_cond_elem = self.get_xml_element('vol_cond')
            vol_cond_elem.TargetVolume = 2.0 * float(vol_cond_elem.TargetVolume)

