from PySteppables import *
import CompuCell
import sys


class SteeringVolumeFlexSteppable(SteppableBasePy):
    def __init__(self, _simulator, _frequency=1):
        SteppableBasePy.__init__(self, _simulator, _frequency)

    def start(self):
        pass

    def step(self, mcs):
        """
        Here is an example of how to modify values stored in the CC3D XML:
        1. First we need to construct the access path that can get us from a room XML element <CompuCell3D>
        to the desired element. Access path is a list of lists where each "inner list" recursively identifies XML elements
        Here is an example:
        lets identify the access path for the following hierarchy of XML elements

        <CompuCell3D>
         <Plugin Name="Volume">
          <VolumeEnergyParameters CellType="Condensing" LambdaVolume="2.0" TargetVolume="25"/>

        to get from the  top element i.e <CompuCell3D>
        its child i.e. <Plugin Name="Volume">
        we insert a list of element identifiers that uniquely identify the required element
        this list has the following format:
        [element_name, attribute_label_1, attribute_value_1, attribute_label_2, attribute_value_2,...]

        IMPORTANT: all elements of the above list are strings

        In out case it is
        ['Plugin','Name','Volume'] - because 'Plugin' is a name of the element , 'Name' is the label of first attribute and
        'Volume' is the value of the first attribute. NOtice tha this uniquely identifies element in the xml .
        In other words, you cannot find another element in the XML that matches this description
        so the access path to

        <CompuCell3D>
         <Plugin Name="Volume">

        looks as follows

        access_path = [['Plugin','Name','Volume']]

        It is a list of lists and the inner list identifies a child of the root element. Notice that we do not list
        root element in the access path.

        Now let's go and identify, secend component of the access path that will "extend" the access path and take use from
        <CompuCell3D>
         <Plugin Name="Volume">

        to

        <CompuCell3D>
         <Plugin Name="Volume">
          <VolumeEnergyParameters CellType="Condensing" LambdaVolume="2.0" TargetVolume="25"/>


       To do that we have to append an list of identifiers that will uniquely identify

       <VolumeEnergyParameters CellType="Condensing" LambdaVolume="2.0" TargetVolume="25"/>
       among the child elements of

       <Plugin Name="Volume">

       Notice that there are two such chile elements
          <VolumeEnergyParameters CellType="Condensing" LambdaVolume="2.0" TargetVolume="25"/>
          <VolumeEnergyParameters CellType="NonCondensing" LambdaVolume="2.0" TargetVolume="25"/>

        The question is what is the sequence of identifiers of the form
        [element_name, attribute_label_1, attribute_value_1, attribute_label_2, attribute_value_2,...]

        that will identify
        <VolumeEnergyParameters CellType="Condensing" LambdaVolume="2.0" TargetVolume="25"/>

        Here it is:
        ['VolumeEnergyParameters', 'CellType', 'Condensing']
        'VolumeEnergyParameters' is the name of the xml element, 'CellType' is a label of first attribute
        and 'Condensing' is the value of the first attribute. Notice that this is sufficient to uniquely identify
        <VolumeEnergyParameters CellType="Condensing" LambdaVolume="2.0" TargetVolume="25"/>

        Now we extend our access path
        access_path = [['Plugin','Name','Volume'], ['VolumeEnergyParameters', 'CellType', 'Condensing']]

        At this point we can use this path to access and modify XML elements
        for read the value of the attribute of element specified by the acces path we use
        getXMLAttributeValue function with the following syntax:

        self.getXMLAttributeValue (attribute_name, *access_path)

        Note the '*' operator in front of access path. This is, so called, unpacking operator and all it does
        it "transforms" a Python list into a sequence of function call parameters and it is a python way to
        declare a function with a variable list of parameters. Ini our case self.getXMLAttributeValue expects first
        argument to be a name of the attribute and followed by comma-separated sequence of access path elements
        i.e.

        to get attribute 'TargetVolume' we call the function as follows

        target_volume_value_str = self.getXMLAttributeValue ('TargetVolume', ['Plugin','Name','Volume'], ['VolumeEnergyParameters', 'CellType', 'Condensing'])

        Notice that w took our access path and used a sequence of inner lists as argumants of the function.
        So why did we bother constructing a list of lists?

        The answer is that we can do the following:

        access_path = [['Plugin','Name','Volume'], ['VolumeEnergyParameters', 'CellType', 'Condensing']]

        target_volume_value_str = self.getXMLAttributeValue ('TargetVolume', *access_path)

        Notice that now the function call is much simpler and easier to read . you construct a path first and then
        can use it multiple times through out the code without being too verbose.

        Notice that the output of fetching the attribute is a string because everything in the XML is a string
        If w want to fo arithmetic operations on a number repersented by strings we need to convert is
        to an appropriate type. It is easy in python - just use type conversion operator

        target_volume_value = float(self.getXMLAttributeValue ('TargetVolume', *access_path))

        now, if you want ot set the value of the XML element specified by the access path all you do is call

        self.setXMLAttributeValue function with the following syntax

        self.setXMLAttributeValue (attribute_name, attribute_value, *access_path)

        Here is a full  example:

        access_path = [['Plugin','Name','Volume'], ['VolumeEnergyParameters', 'CellType', 'Condensing']]
        target_volume_value = float(self.getXMLAttributeValue ('TargetVolume', *access_path))
        self.setXMLAttributeValue ('TargetVolume', 2*target_volume_value, *access_path)

        If you want to access (and change) a value of the element (not the attribute)

        e.g.

        <CompuCell3D>
          <Potts>
            <Temperature>10</Temperature>


        you have to use getXMLElementValue/setXMLElementValue functions

        Here is how we do it:

            temperature_access_path = [['Potts'], ['Temperature']]
            temp = float(self.getXMLElementValue(*temperature_access_path))
            self.setXMLElementValue(temp + 10, *temperature_access_path)

        Notice that temperature_access_path = [['Potts'], ['Temperature']] specifies access path to the <Temperature>
        element. Why? <Potts> element is a child of the <CompuCell3D> element (remember that we do not list root element)
        and <Temperature> is a child of the <Potts> element hence the access path contains only two inner list
        one that identifies <Potts> and one that identifies its child -  <Temperature>

        [['Potts'], ['Temperature']]


        IMPORTANT:
        After you change the values in the XML elements you have to call

        self.updateXML()

        to make sure that CC3D reinterprets XML and applies new values to the running simulation


        """
        if mcs == 10:
            access_path = [['Plugin', 'Name', 'Volume'], ['VolumeEnergyParameters', 'CellType', 'Condensing']]
            target_volume_value = float(self.getXMLAttributeValue('TargetVolume', *access_path))
            self.setXMLAttributeValue('TargetVolume', 2 * target_volume_value, *access_path)

            temperature_access_path = [['Potts'], ['Temperature']]
            temp = float(self.getXMLElementValue(*temperature_access_path))
            self.setXMLElementValue(temp + 10, *temperature_access_path)

            self.updateXML()

    def finish(self):
        # Finish Function gets called after the last MCS
        pass
