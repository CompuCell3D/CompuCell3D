"""

    TO DO:

    * Keyboard events - Del

    * New Simulation wizard

    * resource properties display

    * 

"""

"""

Module used to link Twedit++5 with CompuCell3D.

"""

# Start-Of-Header

name = "CC3D Python Helper Plugin"
author = "Maciej Swat"
autoactivate = True
deactivateable = True
version = "0.9.0"
className = "CC3DPythonHelper"
packageName = "__core__"
shortDescription = "Plugin which assists with CC3D Python scripting"
longDescription = """This plugin provides provides users with CC3D Python code snippets - making Python scripting in CC3D more convenient."""

# End-Of-Header

from PyQt5.QtWebKitWidgets import QWebView

from cc3d.doc.code_ref.user.build import build as code_ref_build_user
from cc3d.doc.code_ref.developer.build import build as code_ref_build_dev

from cc3d.twedit5.Plugins.TweditPluginBase import TweditPluginBase
from cc3d.twedit5.twedit.utils.global_imports import *
from cc3d.twedit5.Plugins.CC3DPythonHelper.Configuration import Configuration
import os.path
import shutil
from cc3d.twedit5.Plugins.PluginUtils.SnippetMenuParser import SnippetMenuParser
from cc3d.twedit5.Plugins.CC3DPythonHelper.sbmlloaddlg import SBMLLoadDlg
import re
from typing import Optional, Type

html_man_filename_user = os.path.join(code_ref_build_user.man_build_dir, "html", "index.html")
html_man_filename_dev = os.path.join(code_ref_build_dev.man_build_dir, "html", "index.html")

error = ''


class CC3DAPIDocViewerWidget(QWebView):
    """Base API doc viewer"""

    def check_build(self) -> None:
        """Checks for built documentation and builds if necessary"""
        if not os.path.isfile(path=self.html_man_filename_root):
            self.build_docs()

    def load_local(self):
        """Loads from local html source"""
        self.load(QUrl().fromLocalFile(self.html_man_filename_root))

    @property
    def html_man_filename_root(self) -> str:
        """Absolute path to root local documentation"""
        raise NotImplementedError

    def build_docs(self) -> None:
        """Builds html documentation"""
        raise NotImplementedError


class CC3DAPIDocViewerWidgetUser(CC3DAPIDocViewerWidget):
    """User API doc viewer"""

    @property
    def html_man_filename_root(self) -> str:
        return html_man_filename_user

    def build_docs(self) -> None:
        code_ref_build_user.build(builder="html")


class CC3DAPIDocViewerWidgetDev(CC3DAPIDocViewerWidget):
    """Developer API doc viewer"""

    @property
    def html_man_filename_root(self) -> str:
        return html_man_filename_dev

    def build_docs(self) -> None:
        code_ref_build_dev.build(builder="html")


class CC3DAPIDocDockWidget(QDockWidget):
    """Base API doc viewer dock"""

    def __init__(self, _cc3d_py_helper, _parent=None):
        QDockWidget.__init__(self, _parent)
        self.cc3d_py_helper: CC3DPythonHelper = _cc3d_py_helper
        self.viewer_widget = self._viewer_widget_cls(parent=self.parent())
        self.setWidget(self.viewer_widget)

        self._loaded = False

    def show(self) -> None:
        """Handles document building and loading as necessary to perform show"""

        # First-time called after initialization: load docs for viewing
        if not self._loaded:
            self._loaded = True

            if not os.path.isfile(self.viewer_widget.html_man_filename_root):
                # No docs found: build on-demand and load afterward
                self.setWindowTitle(self.window_title + ' - Building docs...')
                t = CC3DAPIDocBuilder(dock=self)
                t.finished.connect(self.viewer_widget.load_local)
                t.finished.connect(lambda: self.setWindowTitle(self.window_title))
                t.start()

            else:
                # Docs found: load now
                self.viewer_widget.load_local()

        super().show()

    def closeEvent(self, ev):
        """Hides panel, rather than closes"""
        self._hide_self()
        ev.ignore()

    def _hide_self(self):
        """Issues call to parent to hide self"""
        raise NotImplementedError

    @property
    def _viewer_widget_cls(self) -> Type[CC3DAPIDocViewerWidget]:
        """Returns the viewer widget class associated with this dock"""
        raise NotImplementedError

    @property
    def window_title(self) -> str:
        """Returns the window title string"""
        raise NotImplementedError


class CC3DAPIDocDockWidgetUser(CC3DAPIDocDockWidget):
    """User API doc viewer dock"""

    def _hide_self(self):
        self.cc3d_py_helper.show_document_panel_user(False)

    @property
    def _viewer_widget_cls(self) -> Type:
        return CC3DAPIDocViewerWidgetUser

    @property
    def window_title(self) -> str:
        return "CC3D User API Docs"


class CC3DAPIDocDockWidgetDev(CC3DAPIDocDockWidget):
    """Developer API doc viewer dock"""

    def _hide_self(self):
        self.cc3d_py_helper.show_document_panel_dev(False)

    @property
    def _viewer_widget_cls(self) -> Type:
        return CC3DAPIDocViewerWidgetDev

    @property
    def window_title(self) -> str:
        return "CC3D Developer API Docs"


class CC3DAPIDocBuilder(QThread):

    def __init__(self, dock: CC3DAPIDocDockWidget):
        super().__init__(parent=dock)
        self.build_docs = dock.viewer_widget.build_docs

    def run(self):
        self.build_docs()


class CC3DPythonHelper(QObject, TweditPluginBase):
    """

    Class implementing the About plugin.

    """

    def __init__(self, ui):

        """

        Constructor

        

        @param ui reference to the user interface object (UI.UserInterface)

        """

        QObject.__init__(self, ui)
        TweditPluginBase.__init__(self)

        self.__ui = ui

        self._cc3d_apidoc_dock_user: Optional[CC3DAPIDocDockWidgetUser] = None
        self._cc3d_apidoc_dock_dev: Optional[CC3DAPIDocDockWidgetDev] = None

        self.configuration = Configuration(self.__ui.configuration.settings)

        self.actions = {}

        self.initialize()

        # useful regular expressions
        self.nonwhitespaceRegex = re.compile('^[\s]*[\S]+')

        self.commentRegex = re.compile('^[\s]*#')

        self.defFunRegex = re.compile('^[\s]*def')

        # block statement - : followed by whitespaces at the end of the line
        self.blockStatementRegex = re.compile(':[\s]*$')

        # block statement - : followed by whitespaces at the end of the line
        self.blockStatementWithCommentRegex = re.compile(':[\s]*[#]+[\s\S]*$')

        # line with comment at the end first group matches anythin except '#' the remaining
        # group catches the rest of the line
        self.lineWithCommentAtTheEndRegex = re.compile('([^#]*)([\s\S]*)')

        self.skipCommentsFlag = False

    def initialize(self):

        '''  

            initializes containers used in the plugin

            '''

        self.actions = {}

        self.actionGroupDict = {}

        self.actionGroupMenuDict = {}

        self.cppMenuAction = None

    def addSnippetDictionaryEntry(self, _snippetName, _snippetProperties):

        self.snippetDictionary[_snippetName] = _snippetProperties

    def getUI(self):

        return self.__ui

    def activate(self):

        """

        Public method to activate this plugin.

        

        @return tuple of None and activation status (boolean)

        """

        self.snippetMapper = QSignalMapper(self.__ui)

        self.snippetMapper.mapped[str].connect(self.__insertSnippet)

        self.__initMenus()

        self.__initActions()

        self.__init_ui()

        return None, True

    def deactivate(self):
        """
        Public method to deactivate this plugin.
        """

        self.snippetMapper.mapped[str].disconnect(self.__insertSnippet)

        for actionName, action in self.actions.items():

            try:
                action.triggered.disconnect(self.snippetMapper.map)
            except TypeError:
                print('Skipping disconnecting from map of action {}'.format(actionName))

        self.cc3dPythonMenu.clear()

        skip_comments_in_python_snippets = self.configuration.setSetting("SkipCommentsInPythonSnippets", self.actions[
            "Skip Comments In Python Snippets"].isChecked())

        self.__ui.menuBar().removeAction(self.cc3dPythonMenuAction)

        self.initialize()

    def __initMenus(self):

        self.cc3dPythonMenu = QMenu("CC3D P&ython", self.__ui.menuBar())

        # inserting CC3D Project Menu as first item of the menu bar of twedit++
        self.cc3dPythonMenuAction = self.__ui.menuBar().insertMenu(self.__ui.fileMenu.menuAction(), self.cc3dPythonMenu)

    def __initActions(self):

        """

        Private method to initialize the actions.

        """

        # lists begining of action names which will be grouped
        self.snippetDictionary = {}

        psmp = SnippetMenuParser()

        snippet_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                         'CC3DPythonHelper/Snippets.py.template'))

        psmp.readSnippetMenu(snippet_file_path)

        snippet_menu_dict = psmp.getSnippetMenuDict()

        # print 'snippet menu dict = ',snippetMenuDict

        for menuName, submenuDict in iter(sorted(snippet_menu_dict.items())):

            print('menuName=', menuName)

            group_menu = self.cc3dPythonMenu.addMenu(menuName)

            for subMenuName, snippet_tuple in iter(sorted(submenuDict.items())):
                action = group_menu.addAction(subMenuName)

                # for lookup int he self.snippetDictionary
                action_key = menuName.strip() + ' ' + subMenuName.strip()

                self.snippetDictionary[action_key] = snippet_tuple

                self.actions[action_key] = action

                action.triggered.connect(self.snippetMapper.map)

                self.snippetMapper.setMapping(action, action_key)

        self.actions["Skip Comments In Python Snippets"] = QAction("Skip Comments In Python Snippets", self,

                                                                   shortcut="",

                                                                   statusTip="Skip Comments In Python Snippets")

        self.actions["Skip Comments In Python Snippets"].setCheckable(True)

        flag = self.configuration.setting("SkipCommentsInPythonSnippets")

        self.skipCommentsInPythonSnippets(flag)

        self.actions["Skip Comments In Python Snippets"].setChecked(flag)

        self.actions["Skip Comments In Python Snippets"].triggered.connect(self.skipCommentsInPythonSnippets)

        self.cc3dPythonMenu.addSeparator()

        # ---------------------------------------

        self.cc3dPythonMenu.addAction(self.actions["Skip Comments In Python Snippets"])

        self.cc3dPythonMenu.addSeparator()

        # ---------------------------------------

        self.actions["Code User Reference Panel"] = QAction("Show Code User Reference Panel", self,
                                                            shortcut="",
                                                            statusTip="Show Code User Reference Manual Panel")
        self.cc3dPythonMenu.addAction(self.actions["Code User Reference Panel"])
        self.actions["Code User Reference Panel"].setCheckable(True)
        self.actions["Code User Reference Panel"].triggered.connect(self.show_document_panel_user)

        self.actions["Code Developer Reference Panel"] = QAction("Show Code Developer Reference Panel", self,
                                                                 shortcut="",
                                                                 statusTip="Show Code Developer Reference Manual Panel")
        self.cc3dPythonMenu.addAction(self.actions["Code Developer Reference Panel"])
        self.actions["Code Developer Reference Panel"].setCheckable(True)
        self.actions["Code Developer Reference Panel"].triggered.connect(self.show_document_panel_dev)

    def __init_ui(self):

        self._cc3d_apidoc_dock_user, self._cc3d_apidoc_dock_dev = self.__create_dock_windows()

        self.__setup_dock_window(self._cc3d_apidoc_dock_user, Qt.RightDockWidgetArea)
        self.__setup_dock_window(self._cc3d_apidoc_dock_dev, Qt.RightDockWidgetArea)

        self.show_document_panel_user(False)
        self.show_document_panel_dev(False)

    def __create_dock_windows(self) -> (CC3DAPIDocDockWidgetUser, CC3DAPIDocDockWidgetDev):
        """
        Creates dock windows for displaying APIs.

        :return: the generated dock windows
        :rtype: tuple of CC3DAPIDocDockWidgetUser, CC3DAPIDocDockWidgetDev
        """
        dock_user = CC3DAPIDocDockWidgetUser(self, self.__ui)
        dock_dev = CC3DAPIDocDockWidgetDev(self, self.__ui)
        dock_user.setObjectName(dock_user.window_title)
        dock_dev.setObjectName(dock_dev.window_title)
        return dock_user, dock_dev

    def __setup_dock_window(self, dock: CC3DAPIDocDockWidget, where: int):
        """
        Configures the dock window created with __create_dock_window().

        :param dock: dock to configure
        :type dock: CC3DAPIDocDockWidget
        :param where: dock widget area to place dock (e.g., Qt.RightDockWidgetArea)
        :type where: int
        :return: None
        """
        self.__ui.addDockWidget(where, dock)

        dock.setWindowTitle(dock.window_title)

    def show_document_panel_user(self, _flag: bool):
        """Shows/hides user API panel"""
        if _flag:
            self._cc3d_apidoc_dock_user.show()
        else:
            self._cc3d_apidoc_dock_user.hide()

        self.actions["Code User Reference Panel"].setChecked(_flag)

    def show_document_panel_dev(self, _flag: bool):
        """Shows/hides developer API panel"""
        if _flag:
            self._cc3d_apidoc_dock_dev.show()
        else:
            self._cc3d_apidoc_dock_dev.hide()

        self.actions["Code Developer Reference Panel"].setChecked(_flag)

    def skipCommentsInPythonSnippets(self, _flag):

        self.skipCommentsFlag = _flag

    def __insertSnippet(self, _snippetName):

        snippet_name_str = str(_snippetName)

        text = self.snippetDictionary[str(_snippetName)].snippet_text

        suggested_indent = self.snippetDictionary[str(_snippetName)].suggested_indent

        editor = self.__ui.getCurrentEditor()

        cur_file_name = str(self.__ui.getCurrentDocumentName())

        base_name, ext = os.path.splitext(cur_file_name)

        if ext != ".py" and ext != ".pyw":
            QMessageBox.warning(self.__ui, "Python files only", "Python code snippets work only for Python files")

            return

        cur_line = 0

        cur_col = 0

        if snippet_name_str == "Cell Attributes Add Dictionary To Cells" or snippet_name_str == "Cell Attributes Add List To Cells":

            cur_line, cur_col = self.findEntryLineForCellAttributes(editor)

            if cur_line == -1:
                QMessageBox.warning(self.__ui, "Could not find insert point",

                                    "Could not find insert point for code cell attribute code. "
                                    "Please make sure you are editing CC3D Main Python script")

                return

        elif snippet_name_str.startswith("Bionet Solver 3. Load SBML Model"):

            print('LOADING MODEL')

            current_path = os.path.abspath(os.path.dirname(cur_file_name))

            print('currentPath=', current_path)

            dlg = SBMLLoadDlg(self)

            dlg.setCurrentPath(current_path)

            model_name = 'MODEL_NAME'

            model_nickname = 'MODEL_NICKNAME'

            model_path = 'PATH_TO_SBML_FILE'

            ret = dlg.exec_()

            if ret:

                model_name = str(dlg.modelNameLE.text())

                model_nickname = str(dlg.modelNicknameLE.text())

                model_file_name = os.path.abspath(str(dlg.fileNameLE.text()))

                model_dir = os.path.abspath(os.path.dirname(model_file_name))

                model_path = 'Simulation/' + os.path.basename(model_file_name)

                if model_dir != current_path:  # copy sbml file into simulation directory

                    shutil.copy(model_file_name, current_path)

            text = """

modelName = "%s"

modelNickname  = "%s" # this is usually shorter version version of model name

modelPath="%s"

integrationStep = 0.2

bionetAPI.loadSBMLModel(modelName, modelPath,modelNickname,  integrationStep)

""" % (model_name, model_nickname, model_path)

        elif snippet_name_str.startswith("Extra Fields"):

            # this function potentially inserts new text - will have to get new cursor position after that
            self.includeExtraFieldsImports(editor)

            cur_line, cur_col = editor.getCursorPosition()

        else:

            cur_line, cur_col = editor.getCursorPosition()

        indentation_levels, indent_consistency = self.findIndentationForSnippet(editor, cur_line)

        if suggested_indent >= 0:
            indentation_levels = suggested_indent

        text_lines = text.splitlines(True)

        for i in range(len(text_lines)):

            text_lines[i] = ' ' * editor.indentationWidth() * indentation_levels + text_lines[i]

            try:  # since we dont want twedit to crash when removing coments the code catches all exceptions

                if self.skipCommentsFlag:

                    comment_found = re.match(self.commentRegex, text_lines[i])

                    if comment_found:
                        # if it is 'regular' line we check if this line is beginning of a block statement
                        text_lines[i] = ''

                    else:
                        match = re.match(self.lineWithCommentAtTheEndRegex, text_lines[i])

                        if match:

                            matchGroups = match.groups()

                            if matchGroups[1] != '':
                                text_lines[i] = self.lineWithCommentAtTheEndRegex.sub(r"\1\n", text_lines[i])

            except:

                print('ERROR WHEN REMOVING COMMENTS IN ', text_lines[i])

        indented_text = ''.join(text_lines)

        current_line_text = str(editor.text(cur_line))

        nonwhitespace_found = re.match(self.nonwhitespaceRegex, current_line_text)

        print("currentLineText=", current_line_text, " nonwhitespaceFound=", nonwhitespace_found)

        editor.beginUndoAction()  # begining of action sequence

        if nonwhitespace_found:  # we only add new line if the current line has someting in it other than whitespaces

            editor.insertAt("\n", cur_line, editor.lineLength(cur_line))

            cur_line += 1

        editor.insertAt(indented_text, cur_line, 0)

        # editor.insertAt(text,curLine,0)

        editor.endUndoAction()  # end of action sequence

        # highlighting inserted text

        editor.findFirst(indented_text, False, False, False, True, cur_line)

        line_from, col_from, line_to, col_to = editor.getSelection()

    def includeExtraFieldsImports(self, _editor):

        player_from_import_regex = re.compile('^[\s]*from[\s]*cc3d.*cpp.*PlayerPython[\s]*import[\s]*\*')

        compu_cell_setup_import_regex = re.compile('^[\s]*from[\s]*cc3d[\s]*import[\s]*CompuCellSetup')

        cur_line, cur_col = _editor.getCursorPosition()

        found_player_imports = None

        found_compu_cell_setup_import = None

        for line in range(cur_line, -1, -1):

            text = str(_editor.text(line))

            found_player_imports = re.match(player_from_import_regex, text)

            if found_player_imports:
                break

        for line in range(cur_line, -1, -1):

            text = str(_editor.text(line))

            found_compu_cell_setup_import = re.match(compu_cell_setup_import_regex, text)

            if found_compu_cell_setup_import:
                break

        if not found_compu_cell_setup_import:
            _editor.insertAt("from cc3d import CompuCellSetup\n", 0, 0)

        if not found_player_imports:
            _editor.insertAt("from cc3d.cpp.PlayerPython import * \n", 0, 0)

    def findEntryLineForCellAttributes(self, _editor):

        getCoreSimulationObjectsRegex = re.compile('^[\s]*sim.*CompuCellSetup\.getCoreSimulationObjects')

        text = ''

        found_line = -1

        for line in range(_editor.lines()):

            text = str(_editor.text(line))

            get_core_simulation_objects_regex_found = re.match(getCoreSimulationObjectsRegex,

                                                          text)  # \S - non -white space \swhitespace

            if get_core_simulation_objects_regex_found:  # line with getCoreSimulationObjectsRegex

                found_line = line

                break

        if found_line >= 0:

            # check for comment code  - #add extra attributes here

            attrib_comment_regex = re.compile('^[\s]*#[\s]*add[\s]*extra[\s]*attrib')

            for line in range(found_line, _editor.lines()):

                text = str(_editor.text(line))

                attrib_comment_found = re.match(attrib_comment_regex, text)

                if attrib_comment_found:
                    found_line = line

                    return found_line, 0

            return found_line, 0

        return -1, -1

    def findIndentationForSnippet(self, _editor, _line):

        # nonwhitespaceRegex=re.compile('^[\s]*[\S]+')
        # commentRegex=re.compile('^[\s]*#')
        # defFunRegex=re.compile('^[\s]*def')
        # blockStatementRegex=re.compile(':[\s]*$') # block statement - : followed by whitespaces at the end of the line
        # blockStatementWithCommentRegex=re.compile(':[\s]*[#]+[\s\S]*$') # block statement - :
        # followed by whitespaces at the end of the line

        # ':[\s]*$|:[\s]*[#]+[\s\S*]$'

        # ':[\s]*[\#+[\s\S*]$'

        # ':[\s]*[#]+[\s\S]*' -  works

        text = ''

        for line in range(_line, -1, -1):

            text = str(_editor.text(line))

            nonwhitespace_found = re.match(self.nonwhitespaceRegex, text)  # \S - non -white space \swhitespace

            if nonwhitespace_found:  # once we have line with non-white spaces we check if this is non comment line

                comment_found = re.match(self.commentRegex, text)

                if not comment_found:
                    # if it is 'regular' line we check if this line is beginning of a block statement
                    block_statement_found = re.search(self.blockStatementRegex, text)

                    block_statement_with_comment_found = re.search(self.blockStatementWithCommentRegex, text)

                    if block_statement_found or block_statement_with_comment_found:
                        # we insert code snippet increasing indentation after beginning of block statement

                        indentation_levels = (_editor.indentation(
                            line) + _editor.indentationWidth()) // _editor.indentationWidth()

                        # if this is non-zero indentations in the code are inconsistent
                        indentation_level_consistency = not (_editor.indentation(
                            line) + _editor.indentationWidth()) % _editor.indentationWidth()

                        if not indentation_level_consistency:
                            QMessageBox.warning(self.__ui, "Possible indentation problems",
                                                "Please position code snippet manually using "
                                                "TAB (indent) Shift+Tab (Unindent)")

                            return 0, indentation_level_consistency

                        return indentation_levels, indentation_level_consistency

                    else:
                        # we use indentation of the previous line

                        indentation_levels = (_editor.indentation(line)) // _editor.indentationWidth()

                        # if this is non-zero indentations in the code are inconsistent
                        indentation_level_consistency = not (_editor.indentation(line)) % _editor.indentationWidth()

                        if not indentation_level_consistency:
                            QMessageBox.warning(self.__ui, "Possible indentation problems",
                                                "Please position code snippet manually "
                                                "using TAB (indent) Shift+Tab (Unindent)")

                            return 0, indentation_level_consistency

                        return indentation_levels, indentation_level_consistency

        return 0, 0
