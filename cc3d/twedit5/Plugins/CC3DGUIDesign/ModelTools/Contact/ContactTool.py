# Start-Of-Header

name = 'Contact'

author = 'T.J. Sego'

version = '0.0.0'

class_name = 'ContactTool'

module_type = 'Plugin'

short_description = 'Contact plugin tool'

long_description = """This tool provides model design support for the Contact plugin, including a graphical user 
interface, CC3DML parser and generator, and data validation with CellType plugin"""

tool_tip = """This tool provides model design support for the Contact plugin."""

# End-Of-Header

from collections import OrderedDict
from copy import deepcopy
from itertools import product
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from cc3d.cpp.CC3DXML import *
from cc3d.core.XMLUtils import ElementCC3D, CC3DXMLListPy
from cc3d.core.XMLUtils import dictionaryToMapStrStr as d2mss
from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.CC3DModelToolBase import CC3DModelToolBase
from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.CellType.CellTypeTool import CellTypeTool

from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.Contact.contactdlg import ContactGUI


class ContactTool(CC3DModelToolBase):
    def __init__(self, sim_dicts=None, root_element=None, parent_ui: QObject = None):
        self._dict_keys_to = ['contactMatrix', 'NeighborOrder']
        self._dict_keys_from = ['data', 'NeighborOrder']
        self._requisite_modules = ['Potts', 'CellType']

        self.cell_type_names = None
        self.neighbor_order = 4
        self.contact_matrix = None
        self.user_decision = None

        super(ContactTool, self).__init__(dict_keys_to=self._dict_keys_to, dict_keys_from=self._dict_keys_from,
                                          requisite_modules=self._requisite_modules, sim_dicts=sim_dicts,
                                          root_element=root_element, parent_ui=parent_ui)

    def _process_imports(self) -> None:
        self.cell_type_names = OrderedDict()
        self.contact_matrix = {}

        cell_type_dict = self._sim_dicts['data']
        if cell_type_dict is None:
            return

        type_ids = list(cell_type_dict.keys())
        type_ids.sort()
        for type_id in type_ids:
            self.cell_type_names[type_id] = cell_type_dict[type_id][0]

        if self._sim_dicts['NeighborOrder'] is not None:
            self.neighbor_order = self._sim_dicts['NeighborOrder']

        for t1 in self.cell_type_names.values():
            self.contact_matrix[t1] = {t2: 0.0 for t2 in self.cell_type_names.values()}

        if self._sim_dicts['contactMatrix'] is not None:
            contact_matrix_import = self._sim_dicts['contactMatrix']
        else:
            contact_matrix_import = {}

        for t1, t2 in product(self.cell_type_names.values(), self.cell_type_names.values()):
            try:
                val = contact_matrix_import[t1][t2]
                self.contact_matrix[t1][t2] = val
            except KeyError:
                try:
                    val = contact_matrix_import[t2][t1]
                    self.contact_matrix[t1][t2] = val
                except KeyError:
                    pass

        for t1, t2 in product(self.cell_type_names.values(), self.cell_type_names.values()):
            self.contact_matrix[t2][t1] = self.contact_matrix[t1][t2]

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
        else:
            for type_id, type_tuple in sim_dicts['data'].items():
                if type_id not in self.cell_type_names.keys() or self.cell_type_names[type_id] != type_tuple[0]:
                    return False

            for type_id, type_name in self.cell_type_names.items():
                if type_id not in sim_dicts['data'].keys() or sim_dicts['data'][type_id][0] != type_name:
                    return False

        if 'NeighborOrder' in sim_dicts.keys() and sim_dicts['NeighborOrder'] != self.neighbor_order:
            return False

        if 'contactMatrix' not in sim_dicts.keys() or self.contact_matrix != sim_dicts['contactMatrix']:
            return False

        cell_types = [type_tuple[0] for type_tuple in sim_dicts['data'].values()]
        if any([cell_type for cell_type in cell_types if cell_type not in self.contact_matrix.keys()]):
            return False

        if any([cell_type for cell_type in self.contact_matrix.keys() if cell_type not in cell_types]):
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
        return ElementCC3D('Plugin', {'Name': 'Contact'})

    def generate(self) -> ElementCC3D:
        """
        Generates plugin element from current sim dictionary states
        :return: plugin element from current sim dictionary states
        """
        element = self.get_tool_element()
        for id1 in range(0, self.cell_type_names.keys().__len__()):
            for id2 in range(id1, self.cell_type_names.keys().__len__()):
                try:
                    t1 = self.cell_type_names[id1]
                    t2 = self.cell_type_names[id2]
                    element.ElementCC3D('Energy', {'Type1': t1, 'Type2': t2}, str(self.contact_matrix[t1][t2]))
                except IndexError:
                    pass

        element.ElementCC3D('NeighborOrder', {}, str(self.neighbor_order))

        return element

    def _append_to_global_dict(self, global_sim_dict: dict = None, local_sim_dict: dict = None):
        """
        Public method to append internal sim dictionary; does not call internal update
        :param global_sim_dict: sim dictionary of entire simulation
        :param local_sim_dict: local sim dictionary; default internal dictionary
        :return:
        """

        if global_sim_dict is None:
            global_sim_dict = {}

        if local_sim_dict is None:
            local_sim_dict = deepcopy(self._sim_dicts)

        global_sim_dict['contactMatrix'] = local_sim_dict['contactMatrix']
        global_sim_dict['NeighborOrder'] = local_sim_dict['NeighborOrder']

        return global_sim_dict

    def get_ui(self) -> ContactGUI:
        return ContactGUI(parent=self.get_parent_ui(),
                          contact_matrix=self.contact_matrix,
                          neighbor_order=self.neighbor_order)

    def _process_ui_finish(self, gui: QObject):
        """
        Protected method to process user feedback on GUI close
        :param gui: tool gui object
        :return: None
        """
        self.user_decision = gui.user_decision
        if gui.user_decision:
            self.contact_matrix = gui.contact_matrix
            self.neighbor_order = gui.neighbor_order
            self.__flag_internal_validate = True

    def update_dicts(self):
        """
        Public method to update sim dictionaries from internal data
        :return: None
        """
        self._sim_dicts['NeighborOrder'] = self.neighbor_order
        self._sim_dicts['contactMatrix'] = self.contact_matrix
        return None

    def get_user_decision(self) -> bool:
        return self.user_decision


def load_xml(root_element) -> {}:
    sim_dicts = {}
    for key in ContactTool().dict_keys_from() + ContactTool().dict_keys_to():
        sim_dicts[key] = None

    cell_type_tool = CellTypeTool(root_element=root_element)
    for key, val in cell_type_tool.extract_sim_dicts():
        sim_dicts[key] = val

    plugin_element = root_element.getFirstElement('Plugin', d2mss({'Name': 'Contact'}))

    if plugin_element is None:
        return sim_dicts

    if plugin_element.findElement('NeighborOrder'):
        sim_dicts['NeighborOrder'] = plugin_element.getFirstElement('NeighborOrder').getDouble()

    elements = CC3DXMLListPy(plugin_element.getElements('Energy'))
    contact_matrix_import = {}
    cell_types_import = []
    for element in elements:
        type1 = element.getAttribute('Type1')
        type2 = element.getAttribute('Type2')
        val = element.getDouble()
        if type1 not in contact_matrix_import.keys():
            contact_matrix_import[type1] = {}
        contact_matrix_import[type1][type2] = val

        if type2 not in contact_matrix_import.keys():
            contact_matrix_import[type2] = {}
        contact_matrix_import[type2][type1] = val

        cell_types_import.append(type1)
        cell_types_import.append(type2)

    cell_types = list(set(cell_types_import))
    contact_matrix = {cell_type: {} for cell_type in cell_types}
    for t1, t2 in product(cell_types, cell_types):
        if t1 in contact_matrix_import.keys() and t2 in contact_matrix_import[t1].keys():
            val = contact_matrix_import[t1][t2]
        else:
            val = 0.0
        contact_matrix[t1][t2] = val
        contact_matrix[t2][t1] = val

    sim_dicts['contactMatrix'] = contact_matrix
    return sim_dicts
