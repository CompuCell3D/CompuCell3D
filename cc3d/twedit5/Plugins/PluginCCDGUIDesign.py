# Start-Of-Header

name = "CC3D GUI Design Plugin"

author = "T.J. Sego"

autoactivate = True

deactivateable = True

version = "0.0.0"

className = "CC3DGUIDesign"

packageName = "__core__"

shortDescription = "Plugin that assists with CC3D graphical simulation design"

longDescription = """This plugin provides users with graphical interfaces for model design - 
making CC3D simulation building more convenient."""

# End-Of-Header

"""
Module used to link Twedit++5 with CompuCell3D.
"""
import functools
import inspect
import os
import re
import sys
import traceback
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from cc3d.twedit5.PluginManager.PluginManager import PluginManager
import cc3d.twedit5.Plugins.CC3DMLHelper.SnippetUtils as SnippetUtils
from cc3d.twedit5.Plugins.PluginCCDProject import CC3DProject, CC3DProjectTreeWidget
from cc3d.twedit5.QsciScintillaCustom import QsciScintillaCustom
import cc3d.twedit5.twedit.ActionManager as am
from cc3d.twedit5.twedit.utils import qt_obj_hash
from cc3d.core import XMLUtils
from cc3d.core.CC3DSimulationDataHandler import CC3DSimulationData, CC3DSimulationDataHandler
from cc3d.core.XMLUtils import dictionaryToMapStrStr as d2mss
from cc3d.core.XMLUtils import ElementCC3D, CC3DXMLListPy
from cc3d.cpp.CC3DXML import *
import xml

from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.CC3DModelToolBase import CC3DModelToolBase
from cc3d.twedit5.Plugins.CC3DGUIDesign.CC3DMLCodeScanner import CC3DMLCodeScanner
from cc3d.twedit5.Plugins.CC3DGUIDesign import CC3DMLScannerTools as cc3dst
from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.ModelToolsManager import ModelToolsManager

error = ''


class CC3DGUIDesign(QObject):
    """
    Class implementing the graphical editing plugin.
    """

    def __init__(self, ui):
        """
        Constructor
        :param ui: reference to the user interface object (UI.UserInterface)
        """

        QObject.__init__(self, ui)

        from cc3d.twedit5.EditorWindow import EditorWindow

        self.__ui: EditorWindow = ui

        self.active_editor = None

        self.main_xml_filename = None  # All references to CC3DML file are made to this

        self.main_xml_text = None

        self.model_tools_manager = None

        self.active_tools_dict = {}

        self.active_tools_info = {}

        self.active_tools_action_dict = {}

        self.context_menu_quick_item = {}

        self.context_menu_full_menu = {}

        self.tool_links_dict = {}

        self.active_design_chain = False

        self.begin_line_dict = {}

        self.current_tool = None

        self.sim_dicts = {}

        self.cc3d_xml_to_obj_converter = XMLUtils.Xml2Obj()  # Need this to keep passed CC3DXML element children

        self.code_scanner = CC3DMLCodeScanner(ui=self.get_ui())

        try:

            self.initialize()

        except Exception as e:

            print('Error loading tool: ')

            print(str(e))

            traceback.print_exc(file=sys.stdout)

    def initialize(self):
        """
        Initializes containers used in the plugin
        :return:
        """

        self.model_tools_manager = ModelToolsManager()

        for key, tool, btd in self.model_tools_manager.active_tools():

            self.active_tools_dict[key] = tool

            self.active_tools_info[key] = btd

            begin_line_string, _, begin_line_string_sl = tool().get_enclosing_xml_strings()

            self.begin_line_dict[begin_line_string] = key
            self.begin_line_dict[begin_line_string_sl] = key

        dict_keys_dict = {tool_name: (tool().dict_keys_to(), tool().dict_keys_from())
                          for tool_name, tool in self.active_tools_dict.items()}

        for tool_name in dict_keys_dict.keys():

            self.tool_links_dict[tool_name] = [linked_tool_name
                                               for linked_tool_name in dict_keys_dict.keys() - tool_name
                                               if any(key_to in dict_keys_dict[linked_tool_name][1]
                                                      for key_to in dict_keys_dict[tool_name][0])]

            if self.tool_links_dict[tool_name]:

                print('Established tool link: {} -> {}'.format(str(tool_name), str(self.tool_links_dict[tool_name])))

        self.code_scanner.set_active_tools(active_tools_dict=self.active_tools_dict,
                                           active_tools_info=self.active_tools_info,
                                           tool_links_dict=self.tool_links_dict)

        self.connect_all_signals()

    def activate(self):
        """
        Public method to activate this plugin.
        :return: tuple of None and activation status (boolean)
        """

        self.__init_menus()

        self.__init_actions()

        return None, True

    def deactivate(self):
        """
        Public method to deactivate this plugin.
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def __init_menus():
        """
        Private method to initialize menus
        :return:
        """
        return None

    def __init_actions(self):
        """
        Private method to initialize actions.
        :return:
        """

        for key, tool, btd in self.model_tools_manager.active_tools():

            if tool().get_ui() is not None:

                action_key = str(key) + "Tool"
                qa = QAction(action_key, self)
                qa.setStatusTip(btd.short_description)
                try:
                    qa.triggered.connect(functools.partial(self.on_call_tool, tool))
                except Exception as e:
                    traceback.print_exc(file=sys.stdout)

                self.active_tools_action_dict[key] = {"ActionKey": action_key, "QAction": qa}

                am.addAction(qa)

    def connect_all_signals(self):
        self.__ui.panels[0].currentChanged.connect(self.on_active_editor_change)
        self.__ui.panels[1].currentChanged.connect(self.on_active_editor_change)
        if self.active_editor is not None:
            self.active_editor.cursorPositionChanged.connect(self.on_active_editor_line_change)
            self.active_editor.textChanged.connect(self.on_active_editor_text_change)

    def disconnect_all_signals(self):
        try:
            self.__ui.panels[0].currentChanged.disconnect(self.on_active_editor_change)
        except Exception as e:
            print(str(e))
            pass

        try:
            self.__ui.panels[1].currentChanged.disconnect(self.on_active_editor_change)
        except Exception as e:
            print(str(e))
            pass

        try:
            self.active_editor.cursorPositionChanged.disconnect(self.on_active_editor_line_change)
        except Exception as e:
            print(str(e))
            pass

        try:
            self.active_editor.textChanged.disconnect(self.on_active_editor_text_change)
        except Exception as e:
            print(str(e))
            pass

    def __get_current_main_xml(self):
        """
        Private method to retrieve Twedit current document project main XML script
        :return:
        """

        plugin_manager: PluginManager = self.__ui.pm

        project_plugin: CC3DProject = plugin_manager.getActivePlugin("PluginCCDProject")

        tw: CC3DProjectTreeWidget = project_plugin.treeWidget

        current_tw_item = tw.currentItem()

        tw_project_item = tw.getProjectParent(current_tw_item)

        pdh: CC3DSimulationDataHandler = project_plugin.projectDataHandlers[qt_obj_hash(tw_project_item)]

        csd: CC3DSimulationData = pdh.cc3dSimulationData

        return csd.xmlScript

    def set_current_main_xml(self) -> bool:
        """
        Sets main xml file to currently selected project xml file
        :return:
        """

        try:

            self.set_main_xml(main_xml_filename=self.__get_current_main_xml())

            return True

        except RuntimeError:

            traceback.print_exc(file=sys.stdout)

            return False

    def set_main_xml(self, main_xml_filename: str = None):

        basename, ext = os.path.splitext(main_xml_filename)

        if ext != '.xml':

            raise RuntimeError("GUI Design Error: Attempted assignment of CC3DML main script to non-XML: "
                               + main_xml_filename)

        self.main_xml_filename = main_xml_filename

    def get_ui(self):
        return self.__ui

    def get_current_editor(self):
        return self.__ui.getActiveEditor()

    def get_current_main_xml_editor(self):
        if self.main_xml_filename is None:

            return None

        for editor, entry in self.__ui.fileDict.items():

            editor_file_name = entry[0]

            if editor_file_name == self.main_xml_filename:

                return editor

        return None

    def get_current_panel(self):
        return self.get_editor_panel(self.get_current_editor())

    @staticmethod
    def get_editor_panel(editor):
        try:
            return editor.panel
        except AttributeError:
            return None

    def get_current_file_name(self):
        return self.get_editor_file_name(editor=self.get_current_editor())

    def get_editor_file_name(self, editor):
        try:
            return self.__ui.fileDict[editor][0]
        except (AttributeError, KeyError, IndexError):
            return None

    def set_current_root_element_text(self) -> bool:

        if not self.set_current_main_xml():
            return False

        self.main_xml_text = None

        # Find tab with file matching self.main_xml_filename
        for editor, entry in self.__ui.fileDict.items():

            editor_file_name = entry[0]

            if editor_file_name == self.main_xml_filename:

                self.main_xml_text = str(editor.text())

                return True

        return False

    def get_current_root_element(self):

        try:

            filtered_main_xml_text = self.code_scanner.get_filtered_xml_string(xml_string=self.main_xml_text)

            if filtered_main_xml_text is None:

                return None

            root_element = self.cc3d_xml_to_obj_converter.ParseString(filtered_main_xml_text)

            return root_element

        except xml.parsers.expat.ExpatError as e:

            QMessageBox.critical(self.__ui, "Error Parsing CC3DML file", e.__str__())

            print('GOT PARSING ERROR:', e)

            return None

    def __update_tool_lines(self, editor, model_tool):

        if inspect.isclass(model_tool):

            tool = model_tool()

        else:

            tool = model_tool

        # Reassemble string with indents for editors

        if editor.indentationsUseTabs():

            tab_string = "\t"

        else:

            indentation_width = editor.indentationWidth()

            tab_string = " " * indentation_width

        indent_list = tool.get_xml_indents()

        model_element = tool.generate()

        model_string = model_element.getCC3DXMLElementString()

        model_string_split = model_string.splitlines()

        for line in range(indent_list.__len__()):

            for indent in range(indent_list[line]):

                model_string_split[line] = tab_string + model_string_split[line]

        # Load model string into clipboard for easy pasting

        clipboard = QApplication.clipboard()

        clipboard.setText(model_string)

        # Find current tool element string block
        begin_line, closing_line = self.__find_tool_lines(editor=editor, model_tool=tool)

        editor.beginUndoAction()  # beginning of action sequence

        # Remove old string block and paste new string
        editor.setSelection(begin_line, 0, closing_line + 1, 0)

        editor.replaceSelectedText('\n'.join(model_string_split) + '\n')

        editor.endUndoAction()  # end of action sequence

    def __remove_tool_lines(self, editor, model_tool) -> None:

        begin_line, closing_line = self.__find_tool_lines(editor=editor, model_tool=model_tool)

        editor.beginUndoAction()  # beginning of action sequence

        editor.setSelection(begin_line, 0, closing_line + 1, 0)

        editor.removeSelectedText()

        editor.endUndoAction()  # end of action sequence

    def get_tool_key_from_selection(self, current_line: int = None):
        """
        Public method to find the model tool key of the current editor selection.
        Returns None if editor is not main XML or no enclosing tool found
        :param current_line:{int} Optional current line selection
        :return:
        """

        lines_scanned = 0

        if self.__get_current_main_xml() != self.get_current_file_name():

            print('GUI Design: no current xml for detection')

            return None

        if self.active_editor is None:

            print('GUI Design: no active editor for detection')

            return None

        if current_line is None:

            if not self.active_editor.hasSelectedText():

                print('GUI Design: no available location for detection')

                return None

            current_line, _ = self.active_editor.getCursorPosition()

        print('GUI Design trying detection...')

        while current_line >= 0:

            pretty_text = cc3dst.pretty_text(str(self.active_editor.text(current_line)).strip())

            if pretty_text in self.begin_line_dict.keys():

                print('GUI Design detected model: ' + self.begin_line_dict[pretty_text])

                return self.begin_line_dict[pretty_text]

            elif cc3dst.is_beginning_module_line(pretty_text):

                module_name = cc3dst.get_module_name(pretty_text)

                if module_name is None:

                    module_name = 'Unknown'

                print('GUI Design detected model without a tool: ' + module_name)

                return None

            elif cc3dst.is_closing_module_line(pretty_text) and lines_scanned > 0:

                print('GUI Design detected outside any tool')

                return None

            current_line -= 1

            lines_scanned += 1

        print('GUI Design: no model detected.')

        return None

    def get_tool_key(self, tool):

        for key, val in self.active_tools_dict.items():

            if tool is val:

                return key

        return None

    @staticmethod
    def __find_tool_lines(editor, model_tool) -> [int, int]:
        """
        Finds beginning and ending lines in editor for model tool
        :param editor: main window editor instance
        :param model_tool: class or instance of a model tool
        :return: beginning and ending lines
        """

        if inspect.isclass(model_tool):

            tool = model_tool()

        else:

            tool = model_tool

        begin_line_string, closing_line_string, begin_line_string_sl = tool.get_enclosing_xml_strings()

        module_line_locator_regex = re.compile(begin_line_string)

        module_sl_line_locator_regex = re.compile(begin_line_string_sl)

        module_closing_line_locator_regex = re.compile(closing_line_string)

        begin_line = -1

        closing_line = -1

        for line in range(editor.lines()):

            text = cc3dst.pretty_text(str(editor.text(line)))

            if re.match(module_line_locator_regex, text):

                begin_line = line

            if re.match(module_sl_line_locator_regex, text):

                return line, line + 1

        if begin_line >= 0:

            for line in range(begin_line, editor.lines()):

                if re.match(module_closing_line_locator_regex, cc3dst.pretty_text(str(editor.text(line)))):

                    closing_line = line

                    break

        return begin_line, closing_line

    def generate_context_menu(self):

        if self.active_editor is None:

            return

        self.active_editor: QsciScintillaCustom

        if self.active_editor not in self.context_menu_quick_item.keys():

            self.context_menu_quick_item[self.active_editor] = None

        if self.active_editor.customContextMenu is None:

            menu = QsciScintillaCustom.createStandardContextMenu(self.active_editor)

        else:

            menu = self.active_editor.customContextMenu

            if self.context_menu_quick_item[self.active_editor] is not None:

                menu.removeAction(self.context_menu_quick_item[self.active_editor])

            self.context_menu_quick_item[self.active_editor] = None

            if self.active_editor in self.context_menu_full_menu.keys() and \
                    self.context_menu_full_menu[self.active_editor] is not None:

                menu.removeAction(self.context_menu_full_menu[self.active_editor])

            self.context_menu_full_menu[self.active_editor] = None

        menu.addSeparator()

        if self.current_tool is not None and \
                self.get_tool_key(self.current_tool) in self.active_tools_action_dict.keys():

            qa = am.actionDict[self.active_tools_action_dict[self.get_tool_key(self.current_tool)]["ActionKey"]]

            menu.addAction(qa)

            self.context_menu_quick_item[self.active_editor] = qa

        tool_menu = QMenu('Design Tools', self.active_editor)

        [tool_menu.addAction(am.actionDict[val["ActionKey"]]) for val in self.active_tools_action_dict.values()]

        self.context_menu_full_menu[self.active_editor] = menu.addMenu(tool_menu)

        menu.installEventFilter(CustomContextMenuEventFilter(editor=self.active_editor,
                                                             show_fcn=functools.partial(
                                                                 self.__add_check_marks_to_context_menu)))

        self.active_editor.registerCustomContextMenu(menu)

    def __add_check_marks_to_context_menu(self):
        """
        Private method to show which tools are present with check marks
        :return: None
        """

        current_main_xml_editor = self.get_current_main_xml_editor()

        if current_main_xml_editor is not None:

            if self.current_tool is not None and self.active_editor in self.context_menu_quick_item.keys() and \
                    self.context_menu_quick_item[self.active_editor] is not None:

                line_b, line_c = self.__find_tool_lines(current_main_xml_editor, self.current_tool)

                self.context_menu_quick_item[self.active_editor].setCheckable(True)

                self.context_menu_quick_item[self.active_editor].setChecked(line_b >= 0 and line_c >= 0)

            for key, qa in self.active_tools_action_dict.items():

                qa["QAction"].setCheckable(True)

                line_b, line_c = self.__find_tool_lines(current_main_xml_editor, self.active_tools_dict[key])

                qa["QAction"].setChecked(line_b >= 0 and line_c >= 0)

    def on_active_editor_change(self):

        try:
            self.active_editor.cursorPositionChanged.disconnect(self.on_active_editor_line_change)
        except Exception as e:
            print(str(e))

        try:
            self.active_editor.textChanged.disconnect(self.on_active_editor_text_change)
        except Exception as e:
            print(str(e))

        self.active_editor = self.get_current_editor()

        if self.active_editor is not None:

            self.active_editor.textChanged.connect(self.on_active_editor_text_change)

            self.active_editor.cursorPositionChanged.connect(self.on_active_editor_line_change)

            self.generate_context_menu()

            self.set_current_root_element_text()

    def on_active_editor_line_change(self, current_line: int, index: int):

        tool_key = self.get_tool_key_from_selection(current_line=current_line)

        if tool_key is None:

            self.current_tool = None

        else:

            self.current_tool = self.active_tools_dict[tool_key]

        self.generate_context_menu()

    def on_active_editor_text_change(self):

        self.generate_context_menu()

        current_file_name = self.get_current_file_name()

        if current_file_name is not None and current_file_name == self.main_xml_filename:

            self.set_current_root_element_text()

    def on_call_tool(self, _model_tool):

        root_element = self.get_current_root_element()

        if root_element is not None:

            self.do_design_chain(starting_tool=_model_tool, root_element=root_element)

    def do_design_chain(self, starting_tool, root_element=None):

        if root_element is None:

            root_element = self.get_current_root_element()

        if root_element is None:

            return

        print('Starting design chain with ' + str(starting_tool))

        sim_dicts = starting_tool(root_element=root_element).get_sim_dicts()

        self.active_design_chain = True

        sim_dicts, tools_wrote = self.design_chain(model_tool=starting_tool, sim_dicts=sim_dicts, tools_wrote={})

        self.active_design_chain = False

        current_main_xml_editor = self.get_current_main_xml_editor()

        if current_main_xml_editor is None:

            return

        [self.__update_tool_lines(editor=current_main_xml_editor, model_tool=tool)
         for tool in tools_wrote.values() if tool.validate_dicts(sim_dicts=sim_dicts)]

        self.set_current_root_element_text()

    def design_chain(self, model_tool, sim_dicts, tools_wrote):

        if not self.active_design_chain:

            print('Returning: ' + str(model_tool))

            return sim_dicts, tools_wrote

        if model_tool in tools_wrote.keys():

            print('Chain link: ' + str(model_tool))

            tool = tools_wrote[model_tool]

            if tool.get_ui() is None or tool.validate_dicts(sim_dicts=sim_dicts):

                return sim_dicts, tools_wrote

        else:

            print('New chain link: ' + str(model_tool))

            sim_dicts_append = model_tool(root_element=self.get_current_root_element()).get_sim_dicts()

            for key in sim_dicts_append.keys() - sim_dicts.keys():

                sim_dicts[key] = sim_dicts_append[key]

            tool = model_tool(sim_dicts=sim_dicts, parent_ui=self.get_ui())

        tool.launch_ui()

        if not tool.get_user_decision():

            print('Exiting chain.')

            self.active_design_chain = False

        elif tool.update_dicts() is None and not tool.validate_dicts(sim_dicts):

            tools_wrote[model_tool] = tool

            for key, val in tool.extract_sim_dicts():

                if val is not None:

                    if key in tool.dict_keys_to():

                        print('   Link sim data: ' + str(key))

                    sim_dicts[key] = val

            tool_key = self.get_tool_key(tool=model_tool)

            if tool_key is not None:

                affected_tools = self.tool_links_dict[tool_key]

                for affected_tool in affected_tools:

                    print('Link connection: ' + str(model_tool) + ' -> ' + str(affected_tool))

                    sim_dicts, model_tools_wrote = self.design_chain(model_tool=self.active_tools_dict[affected_tool],
                                                                     sim_dicts=sim_dicts,
                                                                     tools_wrote=tools_wrote)

        return sim_dicts, tools_wrote


class CustomContextMenuEventFilter(QObject):
    def __init__(self, editor: QsciScintillaCustom, **kwargs):
        super(CustomContextMenuEventFilter, self).__init__(editor)
        self.editor = editor
        self.focus_in_fcn = None
        if 'show_fcn' in kwargs.keys():
            self.show_fcn = kwargs['show_fcn']

    def eventFilter(self, a0: QObject, a1: QEvent) -> bool:
        if a1.type() == QEvent.Show:
            if self.show_fcn is not None:
                self.show_fcn()

        return super(CustomContextMenuEventFilter, self).eventFilter(a0, a1)

