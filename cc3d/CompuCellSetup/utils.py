import cc3d.core.XMLUtils as XMLUtils
import cc3d.CompuCellSetup as CompuCellSetup


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

def getSteppableRegistry():
    """
    returns steppable registry object from persistent globals. Legacy function
    :return: {SeppableRegistry
    """
    return CompuCellSetup.persistent_globals.steppable_registry

