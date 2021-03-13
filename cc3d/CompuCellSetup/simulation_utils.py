from cc3d import CompuCellSetup
from cc3d.core.XMLUtils import CC3DXMLListPy
from pathlib import Path


class CC3DCPlusPlusError(Exception):
    def __init__(self, _message):
        self.message = _message

    def __str__(self):
        return repr(self.message)

def set_output_dir(output_dir: str, abs_path: bool = False) -> None:
    """
    Sets output directory to output_dir. If  abs_path is False
    then the directory path will be w.r.t to workspace directory
    Otherwise it is expected that user provides absolute output path
    :param output_dir: directory name - relative (w.r.t to workspace dir) or absolute
    :param abs_path:  flag specifying if user provided absolute or relative path
    :return:
    """
    pg = CompuCellSetup.persistent_globals
    if abs_path:
        pg.set_output_dir(output_dir=output_dir)
    else:
        pg.set_output_dir(output_dir=str(Path(pg.workspace_dir).joinpath(output_dir)))


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


def extract_type_names_and_ids() -> dict:
    """
    Extracts type_name to type id mapping from CC3DXML
    :return {dict}:
    """

    cc3d_xml2_obj_converter = CompuCellSetup.persistent_globals.cc3d_xml_2_obj_converter
    if cc3d_xml2_obj_converter is None:
        return {}

    type_id_type_name_dict = {}

    plugin_elements = cc3d_xml2_obj_converter.root.getElements("Plugin")

    if len(plugin_elements):
        # handle situation where we have plugins and we are looking for CellType plugin to extract elements
        list_plugin = CC3DXMLListPy(plugin_elements)

        for element in list_plugin:

            if element.getAttribute("Name") == "CellType":
                cell_types_elements = element.getElements("CellType")
                type_id_type_name_dict = extract_type_id_type_name_dict(cell_types_elements=cell_types_elements)
    else:
        # try if <CompuCell3DLatticeData> is available - it will be stored in cc3d_xml2_obj_converter.root

        if cc3d_xml2_obj_converter.root.name == 'CompuCell3DLatticeData':

            cell_types_elements = cc3d_xml2_obj_converter.root.getElements("CellType")
            type_id_type_name_dict = extract_type_id_type_name_dict(cell_types_elements=cell_types_elements)

    return type_id_type_name_dict


def extract_type_id_type_name_dict(cell_types_elements):
    """
    Extract dictionary mapping cell type id to cell type name from a sequence of xml elements that look as follows:
    <CellType TypeId="0" TypeName="Medium"/>
    <CellType TypeId="1" TypeName="Condensing"/>
    <CellType TypeId="2" TypeName="NonCondensing"/>
    Note that this sequence of elements can be found int he CellTypePlugin or in the <CompuCell3DLatticeData> in the
    dml.files
    :param cell_types_elements:
    :return:
    """

    type_id_type_name_dict = {}

    list_cell_type_elements = CC3DXMLListPy(cell_types_elements)
    for cell_type_element in list_cell_type_elements:
        type_name = cell_type_element.getAttribute("TypeName")
        type_id = cell_type_element.getAttributeAsInt("TypeId")
        type_id_type_name_dict[type_id] = type_name

    return type_id_type_name_dict


def check_for_cpp_errors(sim):
    if sim.getRecentErrorMessage() != "":
        raise CC3DCPlusPlusError(sim.getRecentErrorMessage())
