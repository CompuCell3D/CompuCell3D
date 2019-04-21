# import cc3d.core.XMLUtils as XMLUtils
# try:
#     import cc3d.core.XMLUtils as XMLUtils
# except ImportError:
#     pass
from cc3d.core import XMLUtils
# import cc3d.CompuCellSetup as CompuCellSetup
from cc3d import CompuCellSetup
from deprecated import deprecated
from cc3d.core.XMLUtils import ElementCC3D

class XML2ObjConverterAdapter:
    def __init__(self):
        self.root = None
        self.xmlTree = None


def init_modules(sim, _cc3dXML2ObjConverter):
    """

    :param sim:
    :param _cc3dXML2ObjConverter:
    :return:
    """
    pluginDataList = XMLUtils.CC3DXMLListPy(_cc3dXML2ObjConverter.root.getElements("Plugin"))
    for pluginData in pluginDataList:
        print ("Element", pluginData.name)
        sim.ps.addPluginDataCC3D(pluginData)

    steppableDataList = XMLUtils.CC3DXMLListPy(_cc3dXML2ObjConverter.root.getElements("Steppable"))
    for steppableData in steppableDataList:
        print("Element", steppableData.name)
        sim.ps.addSteppableDataCC3D(steppableData)

    pottsDataList = XMLUtils.CC3DXMLListPy(_cc3dXML2ObjConverter.root.getElements("Potts"))
    assert pottsDataList.getBaseClass().size() <= 1, 'You have more than 1 definition of the Potts section'
    if pottsDataList.getBaseClass().size() == 1:
        for pottsData in pottsDataList:
            print("Element", pottsData.name)
            sim.ps.addPottsDataCC3D(pottsData)

    metadataDataList = XMLUtils.CC3DXMLListPy(_cc3dXML2ObjConverter.root.getElements("Metadata"))
    assert metadataDataList.getBaseClass().size() <= 1, 'You have more than 1 definition of the Metadata section'
    if metadataDataList.getBaseClass().size() == 1:
        for metadataData in metadataDataList:
            print("Element", metadataData.name)
            sim.ps.addMetadataDataCC3D(metadataData)

def parseXML( xml_fname):
    """

    :param xml_fname:
    :return:
    """

    cc3dXML2ObjConverter = XMLUtils.Xml2Obj()
    root_element = cc3dXML2ObjConverter.Parse(xml_fname)
    print('root_element=', root_element)
    return cc3dXML2ObjConverter

@deprecated(version='4.0.0', reason="You should use : set_simulation_xml_description")
def setSimulationXMLDescription(_xmlTree):
    """

    :param _xmlTree:
    :return:
    """
    return set_simulation_xml_description(xml_tree=_xmlTree)

def set_simulation_xml_description(xml_tree:ElementCC3D)->None:
    """
    
    :param xml_tree: 
    :return: 
    """
    CompuCellSetup.persistent_globals.cc3d_xml_2_obj_converter = XML2ObjConverterAdapter()
    cc3d_xml_2_obj_converter = CompuCellSetup.persistent_globals.cc3d_xml_2_obj_converter
    cc3d_xml_2_obj_converter.xmlTree = xml_tree
    cc3d_xml_2_obj_converter.root = xml_tree.CC3DXMLElement



def getSteppableRegistry():
    """
    returns steppable registry object from persistent globals. Legacy function
    :return: {SeppableRegistry
    """
    return CompuCellSetup.persistent_globals.steppable_registry

