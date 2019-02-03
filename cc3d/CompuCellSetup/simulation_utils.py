from cc3d import CompuCellSetup

def stop_simulation():
    """
    Stops simulation
    :return:
    """
    CompuCellSetup.persistent_globals.user_stop_simulation_flag = True

# legacy api
stopSimulation = stop_simulation


def extract_lattice_type():
    """
    Fetches lattice type
    :return:
    """
    # global cc3dXML2ObjConverter
    cc3d_xml2_obj_converter = CompuCellSetup.persistent_globals.cc3d_xml_2_obj_converter

    if cc3d_xml2_obj_converter.root.findElement("Potts"):
        # dealing with regular cc3dml
        if cc3d_xml2_obj_converter.root.getFirstElement("Potts").findElement("LatticeType"):
            return cc3d_xml2_obj_converter.root.getFirstElement("Potts").getFirstElement("LatticeType").getText()
    else:
        # dealing with LDF file
        if cc3d_xml2_obj_converter.root.findElement("Lattice"):
            return cc3d_xml2_obj_converter.root.getFirstElement("Lattice").getAttribute('Type')

    return ''
