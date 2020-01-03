# Start-Of-Header

name = 'CellType'

author = 'T.J. Sego'

version = '0.0.0'

class_name = 'CellTypeTool'

module_type = 'Plugin'

short_description = 'CellType plugin tool'

long_description = """This tool provides model design support for the CellType plugin, including a graphical user 
interface and CC3DML parser and generator"""

tool_tip = """This tool provides model design support for the CellType plugin"""

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
from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.CellType.celltypedlg import CellTypeGUI


class CellTypeTool(CC3DModelToolBase):
    def __init__(self, sim_dicts=None, root_element=None, parent_ui: QObject = None):
        self._dict_keys_to = ['data']
        self._dict_keys_from = []
        self._requisite_modules = ['Potts']

        self.cell_type_names = None
        self.cell_type_ids = None
        self.cell_types_frozen = None

        super(CellTypeTool, self).__init__(dict_keys_to=self._dict_keys_to, dict_keys_from=self._dict_keys_from,
                                           requisite_modules=self._requisite_modules, sim_dicts=sim_dicts,
                                           root_element=root_element, parent_ui=parent_ui)

        self._user_decision = True

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
        return ElementCC3D('Plugin', {'Name': 'CellType'})

    def generate(self):
        """
        Generates plugin element from current sim dictionary states
        :return: plugin element from current sim dictionary states
        """
        element = self.get_tool_element()

        for index in range(self.cell_type_ids.__len__()):
            attr = {'TypeId': self.cell_type_ids[index], 'TypeName': self.cell_type_names[index]}
            if self.cell_types_frozen[index]:
                attr['Freeze'] = ""

            element.ElementCC3D('CellType', attr)

        return element

    def _process_imports(self) -> None:
        """
        Performs internal UI processing of dictionary/XML inputs during initialization
        This is where UI internal attributes are initialized, potential disagreements between multiple
        information inputs are reconciled, and default data is set
        :return: None
        """
        if self._sim_dicts is None or not self._sim_dicts:
            return

        self.cell_type_ids = []
        self.cell_type_names = []
        self.cell_types_frozen = []

        cell_type_data = self._sim_dicts['data']
        if cell_type_data is None:
            return
        type_ids = list(cell_type_data.keys())
        type_ids.sort()

        type_id = 0
        for tid in type_ids:
            self.cell_type_ids.append(type_id)
            self.cell_type_names.append(cell_type_data[tid][0])
            self.cell_types_frozen.append(cell_type_data[tid][1])
            type_id += 1

    def validate_dicts(self, sim_dicts=None) -> bool:
        """
        Validates current sim dictionary states against changes
        :param sim_dicts: sim dictionaries with changes
        :return:{bool} valid flag is low when changes in sim_dicts affects UI data
        """
        if sim_dicts is None:
            return True

        new_data = sim_dicts['data']
        current_data = self._sim_dicts['data']

        if new_data is current_data:
            return True
        elif new_data is None and current_data is not None:
            return False
        elif new_data is not None and current_data is None:
            return False
        elif new_data and not current_data:
            return False
        elif not new_data and current_data:
            return False
        else:
            return new_data == current_data

    def _append_to_global_dict(self, global_sim_dict: dict = None, local_sim_dict: dict = None):
        """
        Public method to append internal sim dictionary; does not call internal update
        :param global_sim_dict: sim dictionary of entire simulation
        :param local_sim_dict: local sim dictionary; default internal dictionary
        :return:
        """

        if global_sim_dict is None:
            global_sim_dict = {}

        if local_sim_dict is not None:
            global_sim_dict['data'] = local_sim_dict['data']
        else:
            if self._sim_dicts is None:
                self._sim_dicts = {}
                global_sim_dict['data'] = None

            global_sim_dict['data'] = deepcopy(self._sim_dicts['data'])

        return global_sim_dict

    def get_ui(self):
        """
        Returns UI widget
        :return:
        """
        return CellTypeGUI(cell_types=self.cell_type_names, is_frozen=self.cell_types_frozen)

    def _process_ui_finish(self, gui: CellTypeGUI):
        """
        Protected method to process user feedback on GUI close
        :param gui: tool gui object
        :return: None
        """
        if not gui.user_decision:
            return

        cell_types = gui.cell_types
        is_frozen = gui.is_frozen

        num_old = self.cell_type_names.__len__()
        num_new = cell_types.__len__()

        if not self.cell_type_names:
            if not cell_types:
                self.cell_type_ids = []
                self.cell_type_names = []
                self.cell_types_frozen = []
            else:
                self.cell_type_names = cell_types
                self.cell_types_frozen = is_frozen
                self.cell_type_ids = [i for i in range(num_old)]
            return

        for i in range(num_old):
            if self.cell_type_names[i] not in cell_types:
                self.cell_type_names[i] = None

        for i in range(num_new):
            if cell_types[i] in self.cell_type_names:
                cell_types[i] = None
            else:
                for j in range(num_old):
                    if self.cell_type_names[j] is None:
                        self.cell_type_names[j] = cell_types[i]
                        self.cell_types_frozen[j] = is_frozen[i]
                        cell_types[i] = None
                        break

        idx_keep = [i for i in range(num_old) if self.cell_type_names[i] is not None]
        self.cell_type_names = [self.cell_type_names[i] for i in idx_keep]
        self.cell_types_frozen = [self.cell_types_frozen[i] for i in idx_keep]

        idx_append = [i for i in range(num_new) if cell_types[i] is not None]
        [self.cell_type_names.append(cell_types[i]) for i in idx_append]
        [self.cell_types_frozen.append(is_frozen[i]) for i in idx_append]
        self.cell_type_ids = [i for i in range(self.cell_type_names.__len__())]

    def update_dicts(self):
        """
        Public method to update sim dictionaries from internal data
        :return: None
        """
        self._sim_dicts['data'] = {self.cell_type_ids[i]: (self.cell_type_names[i], self.cell_types_frozen[i])
                                   for i in range(self.cell_type_ids.__len__())}
        return None


def load_xml(root_element) -> {}:
    sim_dicts = {}
    for key in CellTypeTool().dict_keys_from() + CellTypeTool().dict_keys_to():
        sim_dicts[key] = None

    plugin_element = root_element.getFirstElement('Plugin', d2mss({'Name': 'CellType'}))

    if plugin_element is None:
        return sim_dicts

    type_table = {}
    plugin_elements = CC3DXMLListPy(plugin_element.getElements('CellType'))
    for plugin_element in plugin_elements:
        type_id = int(plugin_element.getAttribute('TypeId'))
        type_name = plugin_element.getAttribute('TypeName')
        is_freeze = plugin_element.findAttribute('Freeze')
        type_table[type_id] = (type_name, is_freeze)

    sim_dicts['data'] = type_table

    return sim_dicts
