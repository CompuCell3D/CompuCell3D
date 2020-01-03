# Start-Of-Header

name = 'Potts'

author = 'T.J. Sego'

version = '0.0.0'

class_name = 'PottsTool'

module_type = 'Core'

short_description = 'Cellular Potts Model tool'

long_description = """This tool provides model design support for the Cellular Potts Model, including a graphical user 
interface and CC3DML parser and generator"""

tool_tip = """This tool provides model design support for the Cellular Potts Model."""

# End-Of-Header

from collections import OrderedDict
from itertools import product
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from cc3d.cpp.CC3DXML import *
from cc3d.core.XMLUtils import ElementCC3D, CC3DXMLListPy
from cc3d.core.XMLUtils import dictionaryToMapStrStr as d2mss
from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.CC3DModelToolBase import CC3DModelToolBase
from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.CellType.CellTypeTool import CellTypeTool

from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.Potts.pottsdlg import PottsGUI


class PottsTool(CC3DModelToolBase):
    def __init__(self, sim_dicts=None, root_element=None, parent_ui: QObject = None):
        self._dict_keys_to = ['generalPropertiesData']
        self._dict_keys_from = ['data']
        self._requisite_modules = ['CellType']

        self.dim_x = default_gpd()['Dim'][0]
        self.dim_y = default_gpd()['Dim'][1]
        self.dim_z = default_gpd()['Dim'][2]
        self.membrane_fluctuations = default_gpd()['MembraneFluctuations']
        self.neighbor_order = default_gpd()['NeighborOrder']
        self.mcs = default_gpd()['MCS']
        self.lattice_type = default_gpd()['LatticeType']
        self.bc_x = default_gpd()['BoundaryConditions']['x']
        self.bc_y = default_gpd()['BoundaryConditions']['y']
        self.bc_z = default_gpd()['BoundaryConditions']['z']

        self.cell_types = []
        self.__valid_functions = ['Min', 'Max', 'ArithmeticAverage']

        self.user_decision = None

        super(PottsTool, self).__init__(dict_keys_to=self._dict_keys_to, dict_keys_from=self._dict_keys_from,
                                        requisite_modules=self._requisite_modules, sim_dicts=sim_dicts,
                                        root_element=root_element, parent_ui=parent_ui)

    def _process_imports(self) -> None:
        if 'data' in self._sim_dicts.keys() and self._sim_dicts['data'] is not None:
            self.cell_types = [val[0] for val in self._sim_dicts['data'].values()]
        else:
            self.cell_types = []

        if self._sim_dicts['generalPropertiesData'] is not None:
            gpd = self._sim_dicts['generalPropertiesData']
        else:
            self._sim_dicts['generalPropertiesData'] = {}
            gpd = default_gpd()

        if "Dim" in gpd.keys():
            self.dim_x, self.dim_y, self.dim_z = gpd["Dim"]

        if "MembraneFluctuations" in gpd.keys():
            self.membrane_fluctuations = gpd["MembraneFluctuations"]
            if isinstance(self.membrane_fluctuations, dict):
                cell_types_l = list(self.membrane_fluctuations['Parameters'].keys())
                for cell_type in [cell_type for cell_type in self.cell_types if cell_type not in cell_types_l]:
                    self.membrane_fluctuations['Parameters'][cell_type] = 0

                for cell_type in [cell_type for cell_type in cell_types_l if cell_type not in self.cell_types]:
                    self.membrane_fluctuations['Parameters'].pop(cell_type)

                if 'FunctionName' not in self.membrane_fluctuations.keys() or \
                        self.membrane_fluctuations['FunctionName'] not in self.__valid_functions:
                    self.membrane_fluctuations['FunctionName'] = 'Min'

        if "NeighborOrder" in gpd.keys():
            self.neighbor_order = gpd["NeighborOrder"]

        if "MCS" in gpd.keys():
            self.mcs = gpd["MCS"]

        if "LatticeType" in gpd.keys():
            self.lattice_type = gpd["LatticeType"]

        if "BoundaryConditions" in gpd.keys():
            if 'x' in gpd["BoundaryConditions"].keys():
                self.bc_x = gpd["BoundaryConditions"]['x']

        if "BoundaryConditions" in gpd.keys():
            if 'y' in gpd["BoundaryConditions"].keys():
                self.bc_y = gpd["BoundaryConditions"]['y']

        if "BoundaryConditions" in gpd.keys():
            if 'z' in gpd["BoundaryConditions"].keys():
                self.bc_z = gpd["BoundaryConditions"]['z']

    def validate_dicts(self, sim_dicts=None) -> bool:
        """
        Validates current sim dictionary states against changes
        :param sim_dicts: sim dictionaries with changes
        :return:{bool} valid flag is low when changes in sim_dicts affects UI data
        """
        if sim_dicts is None:
            return True

        if 'data' not in sim_dicts.keys():
            return False

        if 'generalPropertiesData' not in sim_dicts.keys():
            return False
        else:
            gpd = sim_dicts['generalPropertiesData']
            if "Dim" not in gpd.keys() or [self.dim_x, self.dim_y, self.dim_z] != gpd["Dim"]:
                return False

            if "MembraneFluctuations" not in gpd.keys() or self.membrane_fluctuations != gpd["MembraneFluctuations"]:
                return False

            if "NeighborOrder" not in gpd.keys() or self.neighbor_order != gpd["NeighborOrder"]:
                return False

            if "MCS" not in gpd.keys() or self.mcs != gpd["MCS"]:
                return False

            if "LatticeType" not in gpd.keys() or self.lattice_type != gpd["LatticeType"]:
                return False

            if "BoundaryConditions" not in gpd.keys():
                return False

            if 'x' not in gpd["BoundaryConditions"].keys() or self.bc_x != gpd["BoundaryConditions"]['x']:
                return False

            if 'y' not in gpd["BoundaryConditions"].keys() or self.bc_y != gpd["BoundaryConditions"]['y']:
                return False

            if 'z' not in gpd["BoundaryConditions"].keys() or self.bc_z != gpd["BoundaryConditions"]['z']:
                return False

        # If a cell type was added, through validation flag to request user input for parameter IF using a function
        # Otherwise ignore; import processing removes deleted cell types
        if isinstance(self.membrane_fluctuations, dict):
            cell_types = [val[0] for val in sim_dicts['data'].values()]
            if any([cell_type for cell_type in cell_types if cell_type not in self.cell_types]):
                return False

        return True

    def load_xml(self, root_element: CC3DXMLElement) -> None:
        """
        Loads plugin data from root XML element
        :param root_element: root simulation CC3D XML element
        :return: None
        """
        self._sim_dicts = load_xml(root_element=root_element)

    def get_tool_element(self):
        """
        Returns base tool CC3D element
        :return:
        """
        return ElementCC3D('Potts', {})

    def generate(self) -> ElementCC3D:
        """
        Generates plugin element from current sim dictionary states
        :return: plugin element from current sim dictionary states
        """
        gpd = self._sim_dicts['generalPropertiesData']
        element = self.get_tool_element()

        element.addComment("Basic properties of CPM (GGH) algorithm")
        element.ElementCC3D("Dimensions", {"x": gpd["Dim"][0], "y": gpd["Dim"][1], "z": gpd["Dim"][2]})
        element.ElementCC3D("Steps", {}, gpd["MCS"])
        if isinstance(gpd["MembraneFluctuations"], dict):
            mf_element = element.ElementCC3D("FluctuationAmplitude", {})
            for cell_type, param in gpd["MembraneFluctuations"]["Parameters"].items():
                mf_element.ElementCC3D("FluctuationAmplitudeParameters", {"CellType": cell_type,
                                                                          "FluctuationAmplitude": str(param)})

            element.ElementCC3D("FluctuationAmplitudeFunctionName", {}, gpd["MembraneFluctuations"]["FunctionName"])
        else:
            element.ElementCC3D("FluctuationAmplitude", {}, gpd["MembraneFluctuations"])
        element.ElementCC3D("NeighborOrder", {}, gpd["NeighborOrder"])

        if gpd["LatticeType"] != "Square":
            element.ElementCC3D("LatticeType", {}, gpd["LatticeType"])

        if gpd['Dim'][2] > 1:
            dim_list = ['x', 'y', 'z']
        else:
            dim_list = ['x', 'y']
        [element.ElementCC3D('Boundary_' + dim_name, {}, gpd['BoundaryConditions'][dim_name]) for dim_name in dim_list]

        return element

    def _append_to_global_dict(self, global_sim_dict: dict = None, local_sim_dict: dict = None):
        if local_sim_dict is None:
            local_sim_dict = self._sim_dicts

        if global_sim_dict is None:
            global_sim_dict = dict()
            for key in self._dict_keys_to:
                global_sim_dict[key] = {}

        global_sim_dict['generalPropertiesData']['Dim'] = local_sim_dict['generalPropertiesData']['Dim']
        global_sim_dict['generalPropertiesData']['MembraneFluctuations'] = \
            local_sim_dict['generalPropertiesData']['MembraneFluctuations']
        global_sim_dict['generalPropertiesData']['NeighborOrder'] = \
            local_sim_dict['generalPropertiesData']['NeighborOrder']
        global_sim_dict['generalPropertiesData']['MCS'] = local_sim_dict['generalPropertiesData']['MCS']
        global_sim_dict['generalPropertiesData']['LatticeType'] = local_sim_dict['generalPropertiesData']['LatticeType']
        global_sim_dict['generalPropertiesData']['BoundaryConditions'] = \
            local_sim_dict['generalPropertiesData']['BoundaryConditions']

        return global_sim_dict

    def get_ui(self) -> PottsGUI:
        gpd = dict()
        gpd["Dim"] = [self.dim_x, self.dim_y, self.dim_z]
        gpd["MembraneFluctuations"] = self.membrane_fluctuations
        gpd["NeighborOrder"] = self.neighbor_order
        gpd["MCS"] = self.mcs
        gpd["LatticeType"] = self.lattice_type
        gpd["BoundaryConditions"] = OrderedDict()
        gpd["BoundaryConditions"]['x'] = self.bc_x
        gpd["BoundaryConditions"]['y'] = self.bc_y
        gpd["BoundaryConditions"]['z'] = self.bc_z
        return PottsGUI(parent=self.get_parent_ui(), gpd=gpd, cell_types=self.cell_types,
                        valid_functions=self.__valid_functions)

    def _process_ui_finish(self, gui: QObject):
        """
        Protected method to process user feedback on GUI close
        :param gui: tool gui object
        :return: None
        """
        self.user_decision = gui.user_decision
        if gui.user_decision:
            self.dim_x, self.dim_y, self.dim_z = gui.gpd["Dim"]
            self.membrane_fluctuations = gui.gpd["MembraneFluctuations"]
            self.neighbor_order = gui.gpd["NeighborOrder"]
            self.mcs = gui.gpd["MCS"]
            self.lattice_type = gui.gpd["LatticeType"]
            self.bc_x = gui.gpd["BoundaryConditions"]['x']
            self.bc_y = gui.gpd["BoundaryConditions"]['y']
            self.bc_z = gui.gpd["BoundaryConditions"]['z']

    def update_dicts(self):
        """
        Public method to update sim dictionaries from internal data
        :return: None
        """
        self._sim_dicts['generalPropertiesData']["Dim"] = [self.dim_x, self.dim_y, self.dim_z]
        self._sim_dicts['generalPropertiesData']["MembraneFluctuations"] = self.membrane_fluctuations
        self._sim_dicts['generalPropertiesData']["NeighborOrder"] = self.neighbor_order
        self._sim_dicts['generalPropertiesData']["MCS"] = self.mcs
        self._sim_dicts['generalPropertiesData']["LatticeType"] = self.lattice_type
        self._sim_dicts['generalPropertiesData']["BoundaryConditions"] = OrderedDict()
        self._sim_dicts['generalPropertiesData']["BoundaryConditions"]['x'] = self.bc_x
        self._sim_dicts['generalPropertiesData']["BoundaryConditions"]['y'] = self.bc_y
        self._sim_dicts['generalPropertiesData']["BoundaryConditions"]['z'] = self.bc_z
        return None

    def get_user_decision(self) -> bool:
        return self.user_decision


def load_xml(root_element) -> {}:
    sim_dicts = {}

    cell_type_tool = CellTypeTool(root_element=root_element)
    for key, val in cell_type_tool.extract_sim_dicts():
        sim_dicts[key] = val

    gpd = default_gpd()

    potts_element = root_element.getFirstElement('Potts')

    if potts_element is None:
        sim_dicts['generalPropertiesData'] = gpd
        return sim_dicts

    potts_element: CC3DXMLElement
    element: CC3DXMLElement

    element = potts_element.getFirstElement('Dimensions')
    if element:

        if element.findAttribute('x'):
            gpd['Dim'][0] = element.getAttributeAsUInt('x')

        if element.findAttribute('y'):
            gpd['Dim'][1] = element.getAttributeAsUInt('y')

        if element.findAttribute('z'):
            gpd['Dim'][2] = element.getAttributeAsUInt('z')

    element = potts_element.getFirstElement('FluctuationAmplitude')
    if element:
        p_elements = CC3DXMLListPy(element.getElements('FluctuationAmplitudeParameters'))
        if p_elements:
            gpd['MembraneFluctuations'] = {'Parameters': {}, 'FunctionName': 'Min'}
            p_element: CC3DXMLElement
            for p_element in p_elements:
                cell_type = p_element.getAttribute('CellType')
                amp = float(p_element.getAttribute('FluctuationAmplitude'))
                gpd['MembraneFluctuations']['Parameters'][cell_type] = amp

            f_element: CC3DXMLElement = potts_element.getFirstElement('FluctuationAmplitudeFunctionName')
            if f_element:
                gpd['MembraneFluctuations']['FunctionName'] = f_element.getText()

        else:
            gpd['MembraneFluctuations'] = float(element.getText())

    element = potts_element.getFirstElement('Temperature')
    if element:
        gpd['MembraneFluctuations'] = float(element.getText())

    n_element = potts_element.getFirstElement('NeighborOrder')

    if n_element:
        gpd['NeighborOrder'] = float(n_element.getText())

    s_element = potts_element.getFirstElement('Steps')

    if s_element:
        gpd['MCS'] = float(s_element.getText())

    sim_dicts['generalPropertiesData'] = gpd
    return sim_dicts


def default_gpd() -> dict:
    return {'Dim': [256, 256, 1],
            'MembraneFluctuations': 10,
            'NeighborOrder': 1,
            'MCS': 100000,
            'LatticeType': 'Square',
            'BoundaryConditions': {'x': 'NoFlux', 'y': 'NoFlux', 'z': 'NoFlux'}}
