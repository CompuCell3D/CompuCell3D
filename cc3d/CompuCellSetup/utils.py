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
from cc3d.core.PyCoreSpecs import _PyCoreSpecsBase


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
    plugin_data_list = XMLUtils.CC3DXMLListPy(_cc3dXML2ObjConverter.root.getElements("Plugin"))
    for pluginData in plugin_data_list:
        sim.ps.addPluginDataCC3D(pluginData)

    steppable_data_list = XMLUtils.CC3DXMLListPy(_cc3dXML2ObjConverter.root.getElements("Steppable"))
    for steppableData in steppable_data_list:
        sim.ps.addSteppableDataCC3D(steppableData)

    potts_data_list = XMLUtils.CC3DXMLListPy(_cc3dXML2ObjConverter.root.getElements("Potts"))
    assert potts_data_list.getBaseClass().size() <= 1, 'You have more than 1 definition of the Potts section'
    if potts_data_list.getBaseClass().size() == 1:
        for pottsData in potts_data_list:
            sim.ps.addPottsDataCC3D(pottsData)

    metadata_data_list = XMLUtils.CC3DXMLListPy(_cc3dXML2ObjConverter.root.getElements("Metadata"))
    assert metadata_data_list.getBaseClass().size() <= 1, 'You have more than 1 definition of the Metadata section'
    if metadata_data_list.getBaseClass().size() == 1:
        for metadataData in metadata_data_list:
            sim.ps.addMetadataDataCC3D(metadataData)


def parseXML( xml_fname):
    """

    :param xml_fname:
    :return:
    """

    cc3dXML2ObjConverter = XMLUtils.Xml2Obj()
    root_element = cc3dXML2ObjConverter.Parse(xml_fname)
    return cc3dXML2ObjConverter


@deprecated(version='4.0.0', reason="You should use : set_simulation_xml_description")
def setSimulationXMLDescription(_xmlTree):
    """

    :param _xmlTree:
    :return:
    """
    return set_simulation_xml_description(xml_tree=_xmlTree)


def set_simulation_xml_description(xml_tree: ElementCC3D) -> None:
    """
    
    :param xml_tree: 
    :return: 
    """
    CompuCellSetup.persistent_globals.cc3d_xml_2_obj_converter = XML2ObjConverterAdapter()
    cc3d_xml_2_obj_converter = CompuCellSetup.persistent_globals.cc3d_xml_2_obj_converter
    cc3d_xml_2_obj_converter.xmlTree = xml_tree
    cc3d_xml_2_obj_converter.root = xml_tree.CC3DXMLElement


def register_specs(*_specs) -> None:
    """
    Register core specification with CC3D

    :param _specs: variable number of _PyCoreSpecsBase-derived class instances
    :return: None
    """
    for _spec in _specs:
        if not issubclass(type(_spec), _PyCoreSpecsBase):
            raise TypeError("Not a core specs instance")
        CompuCellSetup.persistent_globals.core_specs_registry.register_spec(_spec)
    CompuCellSetup.persistent_globals.core_specs_registry.inject()


def getSteppableRegistry():
    """
    returns steppable registry object from persistent globals. Legacy function
    :return: {SeppableRegistry
    """
    return CompuCellSetup.persistent_globals.steppable_registry

