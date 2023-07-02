from cc3d import CompuCellSetup
from cc3d.core.XMLUtils import CC3DXMLListPy
from pathlib import Path
from typing import List, Dict, Union


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
    Note that this sequence of elements can be found in the CellTypePlugin or in the <CompuCell3DLatticeData> in the
    dml.files

    Note that for elements without an explicit type ID, CellTypePlugin assigns the minimum available integer.

    :param cell_types_elements:
    :return:
    """

    type_id_type_name_dict = {}

    list_cell_type_elements = CC3DXMLListPy(cell_types_elements)

    specified_ids = [e.getAttributeAsInt("TypeId") for e in list_cell_type_elements if e.findAttribute("TypeId")]

    for cell_type_element in list_cell_type_elements:
        type_name = cell_type_element.getAttribute("TypeName")
        if cell_type_element.findAttribute("TypeId"):
            type_id = cell_type_element.getAttributeAsInt("TypeId")
        else:
            available_ids = list(range(len(type_id_type_name_dict.keys()) + len(specified_ids) + 1))
            type_id = min([x for x in available_ids if x not in type_id_type_name_dict.keys()])
        type_id_type_name_dict[type_id] = type_name

    return type_id_type_name_dict


def check_for_cpp_errors(sim):
    if sim and sim.getRecentErrorMessage() != "":
        raise CC3DCPlusPlusError(sim.getRecentErrorMessage())


def str_to_int_container(s: str, container: str = 'list') -> Union[List[str], Dict[str, str]]:
    """
    Converts string - comma separated sequence of integers into list of integers
    :param s:
    :param container:
    :return:
    """

    s = s.replace(" ", "")
    s = s.split(",")

    def val_check(inv_val_str):
        try:
            _ = int(inv_val_str)
        except (ValueError, TypeError):
            return False
        return True

    if container == 'list':
        container_int = [int(val) for val in s if val_check(val)]
    elif container == 'dict':
        container_int = {int(val): int(val) for val in s if val_check(val)}
    else:
        raise TypeError('Container argument can only be "list" or "dict"')

    return container_int


