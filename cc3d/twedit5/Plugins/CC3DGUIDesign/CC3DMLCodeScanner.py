from collections import OrderedDict
import inspect
import traceback
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.Qsci import *
from xml.parsers.expat import ExpatError

from cc3d.core.XMLUtils import Xml2Obj
from cc3d.twedit5.EditorWindow import *
from cc3d.twedit5.QsciScintillaCustom import QsciScintillaCustom

from cc3d.twedit5.Plugins.CC3DGUIDesign import CC3DMLScannerTools as cc3dst
from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.ModelToolsManager import ModelToolsManager
from cc3d.twedit5.Plugins.CC3DGUIDesign.ModelTools.ModelToolsManager import BasicToolData


class CC3DMLCodeScanner:
    def __init__(self, ui: EditorWindow):

        self.editor_dict = {}
        self.__ui = ui

        self.active_tools_dict = {}
        self.active_tools_info = {}
        self.tool_links_dict = {}
        self.xml_to_tools_dict = {}
        self.begin_line_dict = {}
        self.close_line_dict = {}

        self.__initialize()

    def __initialize(self):
        print('Connecting CC3DML Code Scanner...')
        try:
            ui = self.get_ui()
            ui.panels[0].currentChanged.connect(self.on_tab_edit_actions)
            ui.panels[1].currentChanged.connect(self.on_tab_edit_actions)
        except Exception as e:

            print(str(e))

            traceback.print_exc(file=sys.stdout)

    def get_ui(self):
        return self.__ui

    def ui_editors(self):
        for panel in self.get_ui().panels:
            for qw_index in range(panel.count()):
                pw = panel.widget(qw_index)
                if isinstance(pw, QsciScintillaCustom):
                    yield pw

    def get_ui_editors(self):
        return [editor for editor in self.ui_editors()]

    @staticmethod
    def get_editor_panel(editor):
        try:
            return editor.panel
        except AttributeError:
            return None

    def get_editor_file_name(self, editor):
        try:
            return self.__ui.fileDict[editor][0]
        except (AttributeError, KeyError, IndexError):
            return None

    def add_editor(self, editor: QsciScintillaCustom) -> bool:
        if editor in self.editor_dict.keys():
            return False
        editor_file_name = self.get_editor_file_name(editor)
        if editor_file_name is None:
            return False
        _, ext = os.path.splitext(editor_file_name)
        if ext != '.xml':
            return False
        print('CC3DML Code Scanner adding editor ' + str(editor))
        self.editor_dict[editor] = ScannerPack(editor=editor, cs=self)
        return True

    def remove_editor(self, editor: QsciScintillaCustom) -> bool:
        if editor not in self.editor_dict.keys():
            return False
        print('CC3DML Code Scanner dropping editor ' + str(editor))
        self.editor_dict.pop(editor)
        return True

    def on_tab_edit_actions(self, index: int):
        e_to_add = [editor for editor in self.get_ui_editors() if editor not in self.editor_dict.keys()]
        e_to_rem = [editor for editor in self.editor_dict.keys() if editor not in self.get_ui_editors()]
        [self.add_editor(editor) for editor in e_to_add if editor is not None]
        [self.remove_editor(editor) for editor in e_to_rem]

    def set_active_tools(self, active_tools_dict: {} = None, active_tools_info: {} = None, tool_links_dict: {} = None):
        if active_tools_dict is not None:
            self.active_tools_dict = active_tools_dict
            self.set_active_tool_enclosing_strings()
            self.xml_to_tools_dict = {tool().get_xml_name(): tool for tool in active_tools_dict.values()
                                      if tool().get_xml_name() is not None}

        if active_tools_info is not None:
            self.active_tools_info = active_tools_info

        if tool_links_dict is not None:
            self.tool_links_dict = tool_links_dict

    def add_active_tool(self, model_tool_name: str, model_tool) -> bool:
        if model_tool_name in self.active_tools_dict.keys():
            return False
        else:
            self.active_tools_dict[model_tool_name] = model_tool
            if model_tool().get_xml_name() is not None:
                self.xml_to_tools_dict[model_tool().get_xml_name()] = model_tool
            return True

    def remove_active_tool(self, model_tool_name: str = None, model_tool=None) -> bool:
        if model_tool is None:
            if model_tool_name is None or model_tool_name not in self.active_tools_dict.keys():
                return False
            else:
                model_tool = self.active_tools_dict[model_tool_name]

        xml_name = model_tool().get_xml_name()
        if xml_name in self.xml_to_tools_dict.keys():
            self.xml_to_tools_dict.pop(xml_name)

        for key, val in self.active_tools_dict.items():
            if val is model_tool:
                self.active_tools_dict.pop(key)
                return True

        return False

    def set_active_tool_enclosing_strings(self, model_tool_name: str = None, model_tool=None):
        if not self.active_tools_dict:
            return

        if model_tool is None:
            active_tools_dict = self.active_tools_dict
        elif model_tool_name is None:
            return
        else:
            active_tools_dict = {model_tool_name: model_tool}

        for key, tool in active_tools_dict.items():
            begin_line_string, close_line_string, begin_line_string_sl = tool().get_enclosing_xml_strings()

            self.begin_line_dict[begin_line_string] = key
            self.begin_line_dict[begin_line_string_sl] = key
            self.close_line_dict[begin_line_string_sl] = key
            self.close_line_dict[close_line_string] = key

    def get_xml_module_names(self, xml_string: str = None, editor: QsciScintillaCustom = None) -> []:
        if xml_string is None and editor is None:
            return []

        filtered_xml_string = self.get_filtered_xml_string(xml_string=xml_string, editor=editor)
        if filtered_xml_string is None:
            return []

        scanned_blocks = cc3dst.scan_xml_model(filtered_xml_string)
        if not scanned_blocks:
            return []

        sb: cc3dst.ScannedBlock
        return [sb.module_name for sb in scanned_blocks if sb.module_name is not None]

    def requisite_check(self, model_tools=None, xml_string: str = None, editor: QsciScintillaCustom = None) -> {}:
        """
        Public method to return missing requisite modules in xml specs
        :param model_tools: list of model tool objects and/or classes; checks all active if input is None
        :param xml_string: xml model spec
        :param editor: xml model spec editor
        :return: dictionary of missing requisite module names per model tool (empty if all are present)
        """

        missing_requisites = {}

        if xml_string is None and editor is None:
            return missing_requisites
        xml_module_names = self.get_xml_module_names(xml_string=xml_string, editor=editor)
        if not xml_module_names:
            return missing_requisites

        if model_tools is None:
            model_tools = self.active_tools_dict.values()

        for model_tool in model_tools:
            if inspect.isclass(model_tool):
                tool = model_tool
            else:
                tool = model_tool.__class__

            missing_requisites[tool] = [req for req in tool().get_requisite_modules() if req not in xml_module_names]

        return missing_requisites

    def get_scanned_blocks(self, xml_string: str = None, editor: QsciScintillaCustom = None):

        if xml_string is None:

            if editor is None or editor not in self.editor_dict.keys():
                return None

            xml_string = str(editor.text())

        scanned_blocks = cc3dst.scan_xml_model(xml_string)

        if not scanned_blocks:
            return None

        sb: cc3dst.ScannedBlock
        sb_sim_l = [sb for sb in scanned_blocks if sb.module_name == "CompuCell3D"]
        if sb_sim_l.__len__() != 1 or sb_sim_l[0].is_problematic:
            for sb in scanned_blocks:
                sb.is_problematic = True

            return scanned_blocks

        sb_sim: cc3dst.ScannedBlock = sb_sim_l[0]

        to_test_gen = []

        filtered_main_xml_text = self.get_filtered_xml_string(xml_string=xml_string)

        cc3d_xml_to_obj_converter = Xml2Obj()

        for sb in [sb for sb in scanned_blocks if sb is not sb_sim]:
            if sb.beginning_line > sb_sim.closing_line or sb.closing_line < sb_sim.beginning_line:
                sb.set_warn_outside_sim_element()

            try:
                cc3d_xml_to_obj_converter.ParseString("\n".join(sb.block_text))
            except ExpatError as e:
                sb.is_problematic = True
                sb.error_msgs.append(str(e))
                continue

            if sb.module_name in self.xml_to_tools_dict.keys():
                model_tool = self.xml_to_tools_dict[sb.module_name]
                requisite_check = self.requisite_check(model_tools=[model_tool], xml_string=xml_string)
                [sb.set_warn_missing_requisite(msg=missing_req) for missing_req in requisite_check[model_tool]]

                if requisite_check[model_tool].__len__() == 0:
                    to_test_gen.append(sb)

            else:
                to_test_gen.append(sb)

        root_element_x = cc3dst.ElementCC3DX("test_root")
        [root_element_x.ElementCC3DX(cc3d_xml_element=cc3d_xml_to_obj_converter.ParseString("\n".join(sb.block_text)))
         for sb in to_test_gen]
        root_element = root_element_x.CC3DXMLElement

        module_names = [sb.module_name for sb in to_test_gen]

        for sb in to_test_gen:
            if sb.module_name not in self.xml_to_tools_dict.keys():
                continue

            model_tool = self.xml_to_tools_dict[sb.module_name]

            for req in model_tool().get_requisite_modules():
                if req not in module_names:
                    continue

            try:
                model_tool(root_element=root_element)
            except Exception as e:
                sb.is_problematic = True
                sb.error_msgs.append(str(e))

        return scanned_blocks

    def get_sim_element_lines(self, xml_string: str = None, editor: QsciScintillaCustom = None):
        if xml_string is None:

            if editor is None or editor not in self.editor_dict.keys():

                return None

            xml_string = str(editor.text())

        scanned_blocks = cc3dst.scan_xml_model(xml_string)

        if not scanned_blocks:
            return None

        sb_sim_l = [sb for sb in scanned_blocks if sb.module_name == "CompuCell3D"]
        if sb_sim_l.__len__() != 1:
            return -1, -1

        sb_sim = sb_sim_l[0]
        return sb_sim.beginning_line, sb_sim.closing_line

    def get_xml_tags(self, xml_string: str = None, editor: QsciScintillaCustom = None):
        if xml_string is None:

            if editor is None or editor not in self.editor_dict.keys():

                return None

            xml_string = str(editor.text())

        scanned_blocks = cc3dst.scan_xml_model(xml_string)

        if not scanned_blocks:
            return None

        sb: cc3dst.ScannedBlock
        st = CC3DMLScannerTags()
        text_split = xml_string.splitlines()
        tags_inside = [False]*text_split.__len__()
        tags_valid = [0]*text_split.__len__()
        tags_recognized = [False]*text_split.__len__()
        tags = [st.default_tag()]*text_split.__len__()
        for sb in scanned_blocks:

            line_b = max(0, sb.beginning_line)
            line_c = sb.closing_line
            if line_c < 0:
                line_c = text_split.__len__()
            line_c += 1
            num_lines = line_c - line_b

            if sb.module_name == "CompuCell3D":

                if sb.is_problematic:
                    return [st.scanner_tag(is_inside=False, is_valid=False, is_recognized=False)]*text_split.__len__()

                tags_inside[line_b:line_c] = [True]*num_lines
                tags_valid[line_b] = True
                tags_valid[line_c - 1] = True

            else:

                tags_valid[line_b:line_c] = [not sb.is_problematic]*num_lines
                tags_recognized[line_b:line_c] = [sb.module_name in self.active_tools_dict.keys()]*num_lines

        for i in range(tags_valid.__len__()):
            if tags_valid[i] == 0:
                tags_valid[i] = tags_inside[i]

        for i in range(tags_inside.__len__()):
            # noinspection PyTypeChecker
            tags[i] = st.scanner_tag(is_inside=tags_inside[i], is_valid=tags_valid[i], is_recognized=tags_recognized[i])

        return tags

    def get_filtered_xml_string(self, xml_string: str = None, editor: QsciScintillaCustom = None):
        """
        Public method to generate a reliable CC3DML simulation script string
        :return:
        """
        if xml_string is None:

            if editor is None or editor not in self.editor_dict.keys():

                return None

            xml_string = str(editor.text())

        st = CC3DMLScannerTags()
        tags = self.get_xml_tags(xml_string=xml_string)
        text_split = xml_string.splitlines(keepends=True)

        filtered_text_split = []
        for i in range(text_split.__len__()):
            if st.passes_run_filter(tag_string=tags[i]):
                filtered_text_split.append(text_split[i])

        return ''.join(filtered_text_split)

    def show_scanning_status(self, msg: str = None, msg_time: int = 2000):
        ui = self.get_ui()
        if msg is None:
            msg = 'Scanning CC3DML...'
        ui.statusBar().showMessage(msg, msg_time)


class CC3DMLScannerTags:
    _inside = True
    _outside = False
    _valid = True
    _invalid = False
    _recognized = True
    _unrecognized = False
    _tag_dict = {(_inside, _valid, _recognized): "IVR",
                 (_outside, _valid, _recognized): "OVR",
                 (_inside, _invalid, _recognized): "IIR",
                 (_outside, _invalid, _recognized): "OIR",
                 (_inside, _valid, _unrecognized): "IVU",
                 (_outside, _valid, _unrecognized): "OVU",
                 (_inside, _invalid, _unrecognized): "IIU",
                 (_outside, _invalid, _unrecognized): "OIU"}
    _default_tuple = (_outside, _invalid, _unrecognized)
    _default_tag = "OIU"

    def scanner_tag(self, is_inside: bool = True, is_valid: bool = True, is_recognized: bool = True) -> str:
        if is_inside:
            inside = self._inside
        else:
            inside = self._outside
        if is_valid:
            valid = self._valid
        else:
            valid = self._invalid
        if is_recognized:
            recognized = self._recognized
        else:
            recognized = self._unrecognized
        return self._tag_dict[(inside, valid, recognized)]

    def default_tag(self) -> str:
        return self._default_tag

    @staticmethod
    def passes_run_filter(tag_string: str = None, tag_tuple: tuple = None) -> bool:
        if tag_string is not None:
            return tag_string[0] == "I" and tag_string[1] == "V"
        elif tag_tuple is not None:
            return tag_tuple[0] and tag_tuple[1]
        else:
            return False

    def to_string(self, tag_tuple: tuple) -> str:
        try:
            return self._tag_dict[tag_tuple]
        except KeyError:
            return self._default_tag

    def to_tuple(self, tag: str) -> tuple:
        for key, dict_tag in self._tag_dict.items():
            if tag == dict_tag:
                return key
        return self._default_tuple


class CC3DMLCodePainter(QObject):
    col_outside = QColor()
    col_outside.setRgb(178, 178, 178)
    col_invalid = QColor()
    col_invalid.setRgb(255, 167, 169)
    col_valid = QColor()
    col_valid.setRgb(114, 208, 114)
    col_recognized = QColor()
    col_recognized.setRgb(148, 175, 228)
    col_warn = QColor()
    col_warn.setRgb(166, 166, 0)
    col_default = QColor()
    col_default.setRgb(255, 255, 255)

    outside_name = "Outside"

    dwell_time = 2000  # in ms

    dwell_block = pyqtSignal(cc3dst.ScannedBlock)
    end_dwell = pyqtSignal()

    def __init__(self, editor: QsciScintillaCustom, active_tool_names=None):
        super(CC3DMLCodePainter, self).__init__(editor)

        self.editor = editor
        self.scanned_blocks = None
        self.color_blocks = None
        self.active_tool_names = active_tool_names

        self.ind_outside = self.editor.indicatorDefine(self.editor.FullBoxIndicator)
        self.ind_invalid = self.editor.indicatorDefine(self.editor.FullBoxIndicator)
        self.ind_valid = self.editor.indicatorDefine(self.editor.FullBoxIndicator)
        self.ind_recognized = self.editor.indicatorDefine(self.editor.FullBoxIndicator)
        self.ind_warn = self.editor.indicatorDefine(self.editor.FullBoxIndicator)

        self.col_list = [self.col_outside, self.col_invalid, self.col_valid, self.col_recognized, self.col_warn]
        self.ind_list = [self.ind_outside, self.ind_invalid, self.ind_valid, self.ind_recognized, self.ind_warn]
        self.ind_to_msg = {}
        for i in range(self.col_list.__len__()):
            self.editor.setIndicatorForegroundColor(self.col_list[i], self.ind_list[i])
            self.editor.setIndicatorDrawUnder(True, self.ind_list[i])

        self.editor.SendScintilla(self.editor.SCI_SETMOUSEDWELLTIME, self.dwell_time)
        self.editor.SCN_DWELLSTART.connect(self.send_dwell_block)
        self.editor.SCN_DWELLEND.connect(self.end_dwell.emit)

    def blocks_to_colors(self, scanned_blocks):
        sb: cc3dst.ScannedBlock
        self.color_blocks = []
        for sb in scanned_blocks:
            if sb.module_name == "CompuCell3D":
                if sb.is_problematic:
                    line_b = 0
                    line_c = self.editor.lines()
                    self.color_blocks.append((line_b, line_c, self.ind_invalid))
                    return
                else:
                    line_b = 0
                    line_c = sb.beginning_line - 1
                    if line_c >= line_b:
                        self.color_blocks.append((line_b, line_c, self.ind_outside))
                    line_b = min(sb.closing_line + 1, self.editor.lines())
                    line_c = self.editor.lines()
                    if line_c >= line_b:
                        self.color_blocks.append((line_b, line_c, self.ind_outside))
            else:
                if sb.is_problematic:
                    ind = self.ind_invalid
                elif sb.module_name in self.active_tool_names:
                    if not sb.is_missing_requisite():
                        ind = self.ind_recognized
                    else:
                        ind = self.ind_warn
                else:
                    ind = self.ind_valid

                self.color_blocks.append((sb.beginning_line, sb.closing_line, ind))

    def paint_editor(self, scanned_blocks):
        if self.color_blocks is not None:
            self.erase_paint()

        self.scanned_blocks = scanned_blocks
        self.blocks_to_colors(scanned_blocks=scanned_blocks)
        [self.editor.fillIndicatorRange(cb[0], 0, cb[1], self.editor.lineLength(cb[1]) - 1, cb[2])
         for cb in self.color_blocks]
        return

    def erase_paint(self):
        self.scanned_blocks = None
        if self.color_blocks is None:
            return
        for ind in self.ind_list:
            self.editor.clearIndicatorRange(0, 0, self.editor.lines(), 0, ind)
        self.color_blocks = None

    def send_dwell_block(self, position: int, x: int, y: int):
        if self.scanned_blocks is None:
            return
        position = self.editor.lineAt(QPoint(60, y))
        if position < 0:
            return
        sb: cc3dst.ScannedBlock
        for sb in self.scanned_blocks:
            if sb.module_name != "CompuCell3D" and sb.beginning_line <= position <= sb.closing_line:
                self.dwell_block.emit(sb)
                return

        sb = cc3dst.ScannedBlock()
        sb.module_name = self.outside_name
        self.dwell_block.emit(sb)


class CC3DMLHoverMessenger(QObject):

    msg_outside = 'Outside of CC3D simulation element'
    msg_invalid = 'Invalid CC3D module element'
    msg_valid = 'Valid CC3D module element with no model tool'
    msg_recognized = 'Recognized CC3D model tool: '

    def __init__(self, editor: QsciScintillaCustom, active_tools_info: {}, tool_links_dict: {}):
        super(CC3DMLHoverMessenger, self).__init__(editor)

        self.editor = editor
        self.active_tools_info = active_tools_info
        self.tool_links_dict = tool_links_dict

        if not self.editor.hasMouseTracking():
            self.editor.setMouseTracking(True)
        self.editor.mouseMoveEvent = self.mouse_move_capture
        self.mouse_pos = QPoint()

    def mouse_move_capture(self, event: QMouseEvent):
        self.mouse_pos.setX(event.globalX())
        self.mouse_pos.setY(event.globalY())
        super(QsciScintillaCustom, self.editor).mouseMoveEvent(event)

    def show_dwell_msg(self, scanned_block: cc3dst.ScannedBlock):
        if scanned_block.is_problematic:
            msg = self.msg_invalid
            if scanned_block.error_msgs:
                msg += '\n\n' + '\n\n'.join(scanned_block.error_msgs)
        elif scanned_block.module_name == CC3DMLCodePainter.outside_name:
            msg = self.msg_outside
        elif scanned_block.module_name in self.active_tools_info.keys():
            tool_info: BasicToolData = self.active_tools_info[scanned_block.module_name]
            msg = self.msg_recognized + scanned_block.module_name
            msg += '\n'
            msg += tool_info.tool_tip
            if self.tool_links_dict is not None and scanned_block.module_name in self.tool_links_dict.keys():
                tool_links = self.tool_links_dict[scanned_block.module_name]
                if tool_links:
                    msg += '\nDown-stream changes include: ' + ', '.join(tool_links)

            if scanned_block.is_missing_requisite():
                msg += '\nMissing module requisites: ' + ', '.join(scanned_block.msgs_missing_requisite())

        else:
            msg = self.msg_valid
            if scanned_block.module_name is not None:
                msg += ': ' + scanned_block.module_name
        self.update_tool_tip(msg=msg)

    def remove_dwell_msg(self):
        self.update_tool_tip()

    def update_tool_tip(self, msg: str = ''):
        QToolTip.showText(self.mouse_pos, msg)


class ScannerTimer(QTimer):

    draw_delay = int(2000)  # in ms

    def __init__(self, editor: QsciScintillaCustom, cs: CC3DMLCodeScanner):
        super(ScannerTimer, self).__init__(editor)

        self.cs = cs

        self.setSingleShot(True)

    def start(self, draw_delay: int = None) -> None:
        if draw_delay is None:
            draw_delay = self.draw_delay

        self.cs.show_scanning_status(msg='Scanning CC3DML...', msg_time=self.draw_delay)

        super(ScannerTimer, self).start(draw_delay)


class ScannerPack(QObject):
    def __init__(self, editor: QsciScintillaCustom, cs: CC3DMLCodeScanner):
        super(ScannerPack, self).__init__(editor)

        self.editor = editor
        self.cs = cs
        self.cp = CC3DMLCodePainter(editor=self.editor, active_tool_names=list(self.cs.active_tools_dict.keys()))
        self.hm = CC3DMLHoverMessenger(editor=self.editor,
                                       active_tools_info=self.cs.active_tools_info,
                                       tool_links_dict=self.cs.tool_links_dict)
        self.cst = ScannerTimer(editor=self.editor, cs=self.cs)

        self.editor.SCN_FOCUSIN.connect(self.handle_focus_changed)
        self.editor.SCN_FOCUSOUT.connect(self.handle_focus_changed)
        self.cp.dwell_block.connect(self.hm.show_dwell_msg)
        self.cp.end_dwell.connect(self.hm.remove_dwell_msg)
        self.editor.textChanged.connect(self.handle_text_changed)
        self.cst.timeout.connect(self.paint_editor)
        self.cs.get_ui().panels[0].tabBarLocal.currentChanged.connect(self.handle_ui_tab_changed)
        self.cs.get_ui().panels[1].tabBarLocal.currentChanged.connect(self.handle_ui_tab_changed)
        self.cs.get_ui().statusBar().messageChanged.connect(self.handle_status_bar_changed)

        self.is_painted = False
        self.has_warnings = False
        self.has_errors = False

        if self.is_panel_current():
            self.cst.start()

    def is_panel_current(self):
        return self.editor is self.editor.panel.currentWidget()

    def is_ui_active(self):
        return self.editor is self.cs.get_ui().getActiveEditor()

    def handle_text_changed(self):
        # Start time if editor text changed
        self.is_painted = False
        self.cst.start()

    def handle_focus_changed(self):
        # Stop timer if editor is not on top in panel
        if not self.is_panel_current():
            self.cst.stop()
        elif not self.is_painted:
            self.cst.start()

    def handle_ui_tab_changed(self):
        if self.is_ui_active():
            self.post_scanning_status()
        else:
            self.cs.get_ui().statusBar().clearMessage()

    def handle_status_bar_changed(self, msg: str):
        if self.is_ui_active() and msg.__len__() == 0:
            self.post_scanning_status()

    def paint_editor(self):
        scanned_blocks = self.cs.get_scanned_blocks(xml_string=str(self.editor.text()))
        self.has_warnings = False
        self.has_errors = False
        if scanned_blocks is not None and scanned_blocks:
            self.cp.paint_editor(scanned_blocks=scanned_blocks)
            self.is_painted = True

            self.has_warnings = any([sb.is_missing_requisite() for sb in scanned_blocks])
            self.has_errors = any([sb.is_problematic for sb in scanned_blocks])
            self.post_scanning_status()

    def erase_paint(self):
        if self.cp.color_blocks is not None:
            self.cp.erase_paint()
            self.is_painted = False
            self.has_warnings = False
            self.has_errors = False

    def post_scanning_status(self):
        if self.has_errors:
            self.cs.show_scanning_status(msg='CC3DML Code Scanner detected errors.', msg_time=0)
        elif self.has_warnings:
            self.cs.show_scanning_status(msg='CC3DML Code Scanner detected warnings.', msg_time=0)

