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

        self.dim_x = None
        self.dim_y = None
        self.dim_z = None
        self.membrane_fluctuations = None
        self.neighbor_order = None
        self.mcs = None
        self.lattice_type = None
        self.bc_x = None
        self.bc_y = None
        self.bc_z = None
        self.offset = None
        self.k_boltzman = None
        self.anneal = None
        self.flip_to_dim_ratio = None
        self.debug_output_freq = None
        self.random_seed = None
        self.energy_func_calc = None

        self.__load_internals_gpd_from_dict(default_gpd())

        self.cell_types = []
        self.__valid_functions = ['Min', 'Max', 'ArithmeticAverage']

        self.user_decision = None

        super(PottsTool, self).__init__(dict_keys_to=self._dict_keys_to, dict_keys_from=self._dict_keys_from,
                                        requisite_modules=self._requisite_modules, sim_dicts=sim_dicts,
                                        root_element=root_element, parent_ui=parent_ui)

    def __load_internals_gpd_from_dict(self, _gpd) -> None:
        self.dim_x = _gpd['Dim'][0]
        self.dim_y = _gpd['Dim'][1]
        self.dim_z = _gpd['Dim'][2]
        self.membrane_fluctuations = _gpd['MembraneFluctuations']
        self.neighbor_order = _gpd['NeighborOrder']
        self.mcs = _gpd['MCS']
        self.lattice_type = _gpd['LatticeType']
        self.bc_x = _gpd['BoundaryConditions']['x']
        self.bc_y = _gpd['BoundaryConditions']['y']
        self.bc_z = _gpd['BoundaryConditions']['z']
        self.offset = _gpd['Offset']
        self.k_boltzman = _gpd['KBoltzman']
        self.anneal = _gpd['Anneal']
        self.flip_to_dim_ratio = _gpd['Flip2DimRatio']
        self.debug_output_freq = _gpd['DebugOutputFrequency']
        self.random_seed = _gpd['RandomSeed']
        self.energy_func_calc = _gpd['EnergyFunctionCalculator']

    def __get_gpd_from_internals(self) -> dict:
        gpd = dict()
        gpd['Dim'] = [self.dim_x, self.dim_y, self.dim_z]
        gpd['MembraneFluctuations'] = self.membrane_fluctuations
        gpd['NeighborOrder'] = self.neighbor_order
        gpd['MCS'] = self.mcs
        gpd['LatticeType'] = self.lattice_type
        gpd['BoundaryConditions'] = OrderedDict()
        gpd['BoundaryConditions']['x'] = self.bc_x
        gpd['BoundaryConditions']['y'] = self.bc_y
        gpd['BoundaryConditions']['z'] = self.bc_z
        gpd['Offset'] = self.offset
        gpd['KBoltzman'] = self.k_boltzman
        gpd['Anneal'] = self.anneal
        gpd['Flip2DimRatio'] = self.flip_to_dim_ratio
        gpd['DebugOutputFrequency'] = self.debug_output_freq
        gpd['RandomSeed'] = self.random_seed
        gpd['EnergyFunctionCalculator'] = self.energy_func_calc
        return gpd

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

        if "Offset" in gpd.keys():
            self.offset = gpd["Offset"]

        if "KBoltzman" in gpd.keys():
            self.k_boltzman = gpd["KBoltzman"]

        if "Anneal" in gpd.keys():
            self.anneal = gpd["Anneal"]

        if "Flip2DimRatio" in gpd.keys():
            self.flip_to_dim_ratio = gpd["Flip2DimRatio"]

        if "DebugOutputFrequency" in gpd.keys():
            self.debug_output_freq = gpd["DebugOutputFrequency"]

        if "RandomSeed" in gpd.keys():
            self.random_seed = gpd["RandomSeed"]

        if "EnergyFunctionCalculator" in gpd.keys():
            self.energy_func_calc = gpd["EnergyFunctionCalculator"]

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
        elif sim_dicts['generalPropertiesData'] != self.__get_gpd_from_internals():
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

        if gpd["LatticeType"] != default_gpd()["LatticeType"]:
            element.ElementCC3D("LatticeType", {}, gpd["LatticeType"])

        if gpd['Dim'][2] > 1:
            dim_list = ['x', 'y', 'z']
        else:
            dim_list = ['x', 'y']
        [element.ElementCC3D('Boundary_' + dim_name, {}, gpd['BoundaryConditions'][dim_name]) for dim_name in dim_list]

        if gpd['Offset'] != default_gpd()['Offset']:
            element.ElementCC3D('Offset', {}, gpd['Offset'])

        if gpd['KBoltzman'] != default_gpd()['KBoltzman']:
            element.ElementCC3D('KBoltzman', {}, gpd['KBoltzman'])

        if gpd['Anneal'] != default_gpd()['Anneal']:
            element.ElementCC3D('Anneal', {}, gpd['Anneal'])

        if gpd['Flip2DimRatio'] != default_gpd()['Flip2DimRatio']:
            element.ElementCC3D('Flip2DimRatio', {}, gpd['Flip2DimRatio'])

        if gpd['DebugOutputFrequency'] != default_gpd()['DebugOutputFrequency']:
            element.ElementCC3D('DebugOutputFrequency', {}, gpd['DebugOutputFrequency'])

        if gpd['RandomSeed'] != default_gpd()['RandomSeed']:
            element.ElementCC3D('RandomSeed', {}, int(gpd['RandomSeed']))

        if gpd['EnergyFunctionCalculator'] != default_gpd()['EnergyFunctionCalculator']:
            energy_func_calc_element: ElementCC3D = element.ElementCC3D(
                'EnergyFunctionCalculator', {'Type': gpd['EnergyFunctionCalculator']['Type']})
            energy_func_calc_element.ElementCC3D(
                'OutputFileName',
                {'Frequency': gpd['EnergyFunctionCalculator']['OutputFileName']['Frequency']},
                gpd['EnergyFunctionCalculator']['OutputFileName']['OutputFileName']
            )
            ef_dict = {'Frequency': gpd['EnergyFunctionCalculator']['OutputCoreFileNameSpinFlips']['Frequency']}
            if gpd['EnergyFunctionCalculator']['OutputCoreFileNameSpinFlips']['GatherResults']:
                ef_dict['GatherResults'] = ''
            if gpd['EnergyFunctionCalculator']['OutputCoreFileNameSpinFlips']['OutputAccepted']:
                ef_dict['OutputAccepted'] = ''
            if gpd['EnergyFunctionCalculator']['OutputCoreFileNameSpinFlips']['OutputRejected']:
                ef_dict['OutputRejected'] = ''
            if gpd['EnergyFunctionCalculator']['OutputCoreFileNameSpinFlips']['OutputTotal']:
                ef_dict['OutputTotal'] = ''
            energy_func_calc_element.ElementCC3D(
                'OutputCoreFileNameSpinFlips', ef_dict,
                gpd['EnergyFunctionCalculator']['OutputCoreFileNameSpinFlips']['OutputCoreFileNameSpinFlips']
            )

        return element

    def _append_to_global_dict(self, global_sim_dict: dict = None, local_sim_dict: dict = None):
        if local_sim_dict is None:
            local_sim_dict = self._sim_dicts

        if global_sim_dict is None:
            global_sim_dict = dict()
            for key in self._dict_keys_to:
                global_sim_dict[key] = {}

        for key in default_gpd().keys():
            global_sim_dict['generalPropertiesData'][key] = local_sim_dict['generalPropertiesData'][key]

        return global_sim_dict

    def get_ui(self) -> PottsGUI:
        return PottsGUI(parent=self.get_parent_ui(), gpd=self.__get_gpd_from_internals(), cell_types=self.cell_types,
                        valid_functions=self.__valid_functions)

    def _process_ui_finish(self, gui: QObject):
        """
        Protected method to process user feedback on GUI close
        :param gui: tool gui object
        :return: None
        """
        self.user_decision = gui.user_decision
        if gui.user_decision:
            self.__load_internals_gpd_from_dict(gui.gpd)

    def update_dicts(self):
        """
        Public method to update sim dictionaries from internal data
        :return: None
        """
        self._sim_dicts['generalPropertiesData'] = self.__get_gpd_from_internals()
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

    offset_element = potts_element.getFirstElement('Offset')
    if offset_element:
        gpd['Offset'] = float(offset_element.getText())

    coefficient_element = potts_element.getFirstElement('KBoltzman')
    if coefficient_element:
        gpd['KBoltzman'] = float(coefficient_element.getText())

    anneal_element = potts_element.getFirstElement('Anneal')
    if anneal_element:
        gpd['Anneal'] = float(anneal_element.getText())

    flip_to_dim_ratio_element = potts_element.getFirstElement('Flip2DimRatio')
    if flip_to_dim_ratio_element:
        gpd['Flip2DimRatio'] = float(flip_to_dim_ratio_element.getText())

    debug_output_freq_element = potts_element.getFirstElement('DebugOutputFrequency')
    if debug_output_freq_element:
        gpd['DebugOutputFrequency'] = float(debug_output_freq_element.getText())

    random_seed_element = potts_element.getFirstElement('RandomSeed')
    if random_seed_element:
        gpd['RandomSeed'] = int(random_seed_element.getText())

    energy_func_calc_element = potts_element.getFirstElement('EnergyFunctionCalculator')
    if energy_func_calc_element:
        func_type = energy_func_calc_element.getAttribute('Type')

        file_name_element = energy_func_calc_element.getFirstElement('OutputFileName')
        file_name = file_name_element.getText()
        file_freq = float(file_name_element.getAttribute('Frequency'))

        spin_element = energy_func_calc_element.getFirstElement('OutputCoreFileNameSpinFlips')
        spin_name = spin_element.getText()
        spin_freq = float(spin_element.getAttribute('Frequency'))
        gather_results = spin_element.findAttribute('GatherResults')
        output_accepted = spin_element.findAttribute('OutputAccepted')
        output_rejected = spin_element.findAttribute('OutputRejected')
        output_total = spin_element.findAttribute('OutputTotal')

        gpd['EnergyFunctionCalculator'] = {'Type': func_type,
                                           'OutputFileName': {'OutputFileName': file_name,
                                                              'Frequency': file_freq},
                                           'OutputCoreFileNameSpinFlips': {'OutputCoreFileNameSpinFlips': spin_name,
                                                                           'Frequency': spin_freq,
                                                                           'GatherResults': gather_results,
                                                                           'OutputAccepted': output_accepted,
                                                                           'OutputRejected': output_rejected,
                                                                           'OutputTotal': output_total}
                                           }

    sim_dicts['generalPropertiesData'] = gpd
    return sim_dicts


def default_gpd() -> dict:
    return {'Dim': [256, 256, 1],
            'MembraneFluctuations': 10,
            'NeighborOrder': 1,
            'MCS': 100000,
            'LatticeType': 'Square',
            'BoundaryConditions': {'x': 'NoFlux', 'y': 'NoFlux', 'z': 'NoFlux'},
            'Offset': 0.0,
            'KBoltzman': 1.0,
            'Anneal': 0.0,
            'Flip2DimRatio': 1.0,
            'DebugOutputFrequency': None,
            'RandomSeed': None,
            'EnergyFunctionCalculator': None}
