from copy import deepcopy
from PyQt5.Qt import *
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QObject
import re

from cc3d.cpp.CC3DXML import *
from cc3d.core.XMLUtils import ElementCC3D
from cc3d.twedit5.Plugins.CC3DGUIDesign.CC3DMLScannerTools import ElementCC3DX

# Start-Of-Header

name = 'CC3DModelTool'

author = 'T.J. Sego'

version = '0.0.0'

class_name = 'CC3DModelToolBase'

module_type = 'Core'

short_description = 'Superclass for defining CC3D model tools'

long_description = """This superclass defines all requisite functionality for a tool in the GUI Design plugin."""

tool_tip = """This superclass defines all requisite functionality for a tool in the GUI Design plugin."""

# End-Of-Header


class CC3DModelToolBase:
    """
    Template superclass for CC3DML editor tools
    """
    def __init__(self, dict_keys_to: [] = None, dict_keys_from: [] = None, requisite_modules: [] = None, sim_dicts=None,
                 root_element: CC3DXMLElement = None, parent_ui: QObject = None):
        self._xml_element = None
        self._element_cc3d = None
        self._dict_keys_to = dict_keys_to
        self._dict_keys_from = dict_keys_from
        self._requisite_modules = requisite_modules
        self.__parent_ui = parent_ui
        self._user_decision = False

        self.__flag_no_ui = False

        self._sim_dicts = {}
        self.__initialize_sim_containers()
        self.load_sim_dicts(sim_dicts=sim_dicts)
        if root_element is not None:
            self.load_xml(root_element=root_element)

        self._process_imports()
        self.tool_element = self.get_tool_element()

        self.__indent_lvl = -1
        self._indent_list = []

    def dict_keys_to(self):
        """
        Returns keys of affected sim dictionaries
        :return: keys of affected sim dictionaries
        """
        return self._dict_keys_to

    def dict_keys_from(self):
        """
        Returns keys of affecting sim dictionaries
        :return: keys of affecting sim dictionaries
        """
        return self._dict_keys_from

    def get_requisite_modules(self):
        """
        Returns names of required modules by this model tool
        :return:{list} required modules by this model tool
        """
        return self._requisite_modules

    def get_xml_name(self):
        """
        Returns CC3DML name
        :return:
        """
        xml_element = self.get_tool_element().CC3DXMLElement
        if xml_element.findAttribute("Name"):
            return xml_element.getAttribute("Name")
        elif xml_element.findAttribute("Type"):
            return xml_element.getAttribute("Type")
        else:
            return None

    def load_xml(self, root_element: CC3DXMLElement) -> None:
        """
        Loads plugin data from root XML element
        :param root_element: root simulation CC3D XML element
        :return: None
        """
        raise NotImplementedError

    def get_tool_element(self):
        """
        Returns base tool CC3D element
        :return:
        """
        raise NotImplementedError

    def generate(self):
        """
        Generates plugin element from current sim dictionary states
        :return: plugin element from current sim dictionary states
        """
        raise NotImplementedError

    def get_enclosing_xml_strings(self) -> [str, str, str]:
        """
        Returns first and last lines of tool element of multi-line XML spec, and single-line XML spec
        :return:
        """
        tool_element: ElementCC3D = self.get_tool_element()
        element_string: str = tool_element.getCC3DXMLElementString()
        element_string_rsplit = element_string.splitlines()
        if element_string_rsplit.__len__() == 1:
            element_string_sl = element_string_rsplit[0]
            search = re.search('<(.*)/>', element_string_sl)
            if search is None:
                return None, None, None
            m_type = search.group(1).split()[0]
            if search.group(1).split().__len__() == 2:
                content = ' ' + search.group(1).split()[1]
            else:
                content = ''
            element_string_begin = '<' + m_type + content + '>'
            element_string_end = '</' + m_type + '>'
        else:
            element_string_begin = element_string_rsplit[0]
            element_string_end = element_string_rsplit[-1]
            element_string_sl = element_string_rsplit[0]
            search = re.search('<(.*)>', element_string_sl)
            if search is None:
                return None, None, None
            m_type = search.group(1).split()[0]
            element_string_sl = '<' + m_type + '/>'

        return element_string_begin, element_string_end, element_string_sl

    def get_xml_indents(self) -> []:
        """
        Returns list of number of indents for each line of element string
        :return:
        """
        if self._element_cc3d is not None:
            element = self._element_cc3d
        elif self._sim_dicts is not None and self._sim_dicts:
            element = self.generate()
        else:
            element = self.tool_element

        if element is None:
            raise RuntimeError

        element: ElementCC3D
        element_x = ElementCC3DX(cc3d_xml_element=element.CC3DXMLElement)

        self.__indent_lvl = 1
        self._indent_list = []
        self.__iterate_element(element_x=element_x)
        return self._indent_list

    def __iterate_element(self, element_x: ElementCC3DX):
        """
        Private method for recursive indent list assembly
        :param element: Current CC3D element
        :return: None
        """
        self._indent_list.append(self.__indent_lvl)
        self.__indent_lvl += 1
        for child in element_x.element_list:
            self.__iterate_element(element_x=child)
        self.__indent_lvl -= 1
        if element_x.element_list:
            self._indent_list.append(self.__indent_lvl)

    def load_sim_dicts(self, sim_dicts):
        """
        Loads sim dictionaries
        :param sim_dicts: sim dictionaries
        :return: None
        """
        if sim_dicts is None:
            return
        sim_dicts_copy = deepcopy(sim_dicts)
        for key, val in sim_dicts_copy.items():
            if key in self._dict_keys_to + self._dict_keys_from:
                self._sim_dicts[key] = val
        return

    def extract_sim_dicts(self):
        """
        Yields affected sim dictionaries
        :return: affected sim dictionaries
        """
        if self._sim_dicts is None or self._dict_keys_to is None:
            return None
        for key in self._dict_keys_to:
            yield key, self._sim_dicts[key]
        if self._dict_keys_from is not None:
            for key in self._dict_keys_from:
                yield key, self._sim_dicts[key]

    def get_sim_dicts(self):
        """
        Returns affected sim dictionaries
        :return: affected sim dictionaries
        """
        if self.extract_sim_dicts() is None:
            return None
        return deepcopy({key: val for key, val in self.extract_sim_dicts()})

    def __initialize_sim_containers(self):
        """
        Private method initializes None in all internal simulation dictionaries related to this plugin
        :return:
        """
        for key in self._dict_keys_to + self._dict_keys_from:
            if key not in self._sim_dicts.keys():
                self._sim_dicts[key] = None

    def _process_imports(self) -> None:
        """
        Performs internal UI processing of dictionary/XML inputs during initialization
        This is where UI internal attributes are initialized, potential disagreements between multiple
        information inputs are reconciled, and default data is set
        :return: None
        """
        raise NotImplementedError

    def validate_dicts(self, sim_dicts=None) -> bool:
        """
        Validates current sim dictionary states against changes
        :param sim_dicts: sim dictionaries with changes; set to None for internal dictionary validation only
        :return:{bool} valid flag is low when changes in sim_dicts affects UI data
        """
        raise NotImplementedError

    def handle_external_changes(self, sim_dicts) -> bool:
        """
        Processes changes made to sim dictionaries by other tools
        :param sim_dicts: sim dictionaries according to other tools
        :return:{bool} flag is low when external changes affected internal data
        """
        if sim_dicts is None:
            return False

        return_flag = self.validate_dicts(sim_dicts=sim_dicts)

        sim_dicts_local = deepcopy(sim_dicts)
        for key in self._dict_keys_from:
            if key in sim_dicts_local.keys():
                self._sim_dicts[key] = sim_dicts_local[key]
            else:
                self._sim_dicts[key] = None

        self._process_imports()

        return return_flag

    def append_to_global_dict(self, global_sim_dict: dict = None, local_sim_dict: dict = None):
        """
        Public method to append internal sim dictionary; does not call internal update
        :param global_sim_dict: sim dictionary of entire simulation
        :param local_sim_dict: local sim dictionary; default internal dictionary
        :return:
        """
        global_sim_dict_local = deepcopy(global_sim_dict)
        local_sim_dict_local = deepcopy(local_sim_dict)
        if global_sim_dict_local is None:
            global_sim_dict_local = {}

        for key in self._dict_keys_to:
            if key not in global_sim_dict_local.keys():
                global_sim_dict_local[key] = {}

        gs_dict_r = self._append_to_global_dict(global_sim_dict=global_sim_dict_local,
                                                local_sim_dict=local_sim_dict_local)
        return deepcopy(gs_dict_r)

    def _append_to_global_dict(self, global_sim_dict: dict = None, local_sim_dict: dict = None):
        return NotImplementedError

    def get_user_decision(self) -> bool:
        """
        Returns user feedback on UI close
        :return:
        """
        return self._user_decision

    def launch_ui(self) -> None:
        """
        Launches stand-alone GUI
        :return: None
        """
        if self.get_flag_no_ui():
            return None

        gui = self.get_ui()
        if gui is None:
            return None
        qd = self.__setup_standalone(gui=gui)
        qd.exec_()
        self._process_ui_finish(gui=gui)
        return None

    def __setup_standalone(self, gui) -> QDialog:
        qd = QDialog(self.get_parent_ui())
        ql = QGridLayout(qd)
        ql.addWidget(gui)
        qd.setModal(True)
        qd.setWindowTitle(gui.windowTitle())
        gui.setParent(qd)
        gui.mtg_close_signal.connect(qd.close)
        return qd

    def get_ui(self):
        """
        Returns GUI widget, or None if tool does not include a GUI
        :return:
        """
        raise NotImplementedError

    def set_flag_no_ui(self, flag: bool = False):
        """
        Set flag high to not call GUI during call to ui launch
        :param flag:
        :return:
        """
        self.__flag_no_ui = flag

    def get_flag_no_ui(self) -> bool:
        """
        Gets flag to call/not call GUI during call to ui launch
        :return:
        """
        return self.__flag_no_ui

    def _process_ui_finish(self, gui: QObject):
        """
        Protected method to process user feedback on GUI close
        :param gui: tool gui object
        :return: None
        """
        return NotImplementedError

    def get_parent_ui(self):
        """
        Public method to return parent UI
        :return:{QObject} parent ui
        """
        return self.__parent_ui

    def update_dicts(self):
        """
        Public method to update sim dictionaries from internal data
        :return: None
        """
        return NotImplementedError

    def set_parent_ui(self, parent_ui):
        """
        Public method to set parent UI
        :param parent_ui: parent UI object
        :return: None
        """
        if parent_ui is None or isinstance(parent_ui, QObject):
            self.__parent_ui = parent_ui
        else:
            raise ValueError
