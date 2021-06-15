"""
TO DO:
* fix configuration dialog to display loaded plugins
* fix searching in unicode paths - as of now those dirs are reported as non-existent

* cannot find  new line character

* window menu

* help menu

* expand guessLexer function to include more languages

* implement file change autosensing

* list of recent files

* proper command line handling

* replace lock files with named mutex on windows and pid search in unix

* write keyboard shortcut editor widget

* write lexers for additional languages - extend CustomLexer class

* handle undo better

* deal with write protected, hidden, or non-standard files

* have to find better way to check if a given file is open

* saveAs dialog should automatically pick extension based on current extension of edited file

* add printing capabilities

* add EOL conversion

* add text encodings

* add auto completion

* website

* help menu

* find in files should display message - "assembling file list", "searching files" 

* margin click to bookmark/unbookmark

* easy map from/to tab to/from document name 

* remove dbgMsg(statements )

* lexer in find in files has to scan entire text before displaying it - otherwise gui may become unresponsive on fold text 

* change name to twedit++

* cursor change on ? click # done using QtDrawer and setWindowFlags

* add context menu

* check what happens if there is python error during binary file execution - perhaps use try /except in the main script to catch any errors in the

* for new documents save as dialog should open up in directory of the previous widget - if this is not available then default path

* recent file list may be storing too many elements when destroying tabs before exiting

* add new function add new document (it inintializes fileDict. ) and avoid maniulating file dict directly

* figure out how to display binary files including control characters

* add context menu to clear find in files dialog

* add fold all/unfoldall - have to change existing toggle fold all

* python indentation does not work correctly after enter - sohuld assume indent of the previous line - maybe have to do it with API

* NOTICE findNext()

          replace() does not work you have to use findFirst with arguments  

          it als has issues with undo actions so it is best to use findFirst always...

 

* add context menus to tabs (close, delete, rename etc...)

* context menu to for editor

* Default editor should be removed anyway the moment we open any new file

* groups brackets ( ) in regex string have to be escaped - e.g. \(.*\) - have to fix it to make sure you can put regular brackets

* clean up loadFile fcn

* display information if text cannot be found

* list of recent documents

* variables for naming platforms

* Allow new window to be open

* add contex menu to tab - see notepad++ options

 """
# todo - change twedit5.twedit to twedit5.core
# todo - check .encode() calls - is it for unicode?
# todo - fix plugin load / unload dialog in settings
# todo - add warning when cannot detect encoding during opening of the dile

from math import log
import fnmatch
import os
import codecs
import shutil
import re
from cc3d.twedit5.twedit.utils.global_imports import *
from cc3d.twedit5.twedit.utils.string_utils import remove_n_chars
from cc3d.twedit5.twedit.utils.encoding_detector import decode
import cc3d.twedit5.twedit.ActionManager as am
# import application_rc
import cc3d.twedit5.twedit_plus_plus_rc
from cc3d.twedit5.findandreplacedlg import FindAndReplaceDlg
from cc3d.twedit5.findandreplacedlg import FindAndReplaceHistory
from cc3d.twedit5.findandreplacedlg import FindInFilesResults
from cc3d.twedit5.findandreplacedlg import FindDisplayWidget
from cc3d.twedit5.gotolinedlg import GoToLineDlg
from cc3d.twedit5.twedit.editor import Configuration
from cc3d.twedit5.twedit.editor import ConfigurationDlg
from cc3d.twedit5.LanguageManager import LanguageManager
from cc3d.twedit5.CycleTabsPopup import CycleTabsPopup
from cc3d.twedit5.CycleTabsPopup import CycleTabFileList
from cc3d.twedit5.QsciScintillaCustom import QsciScintillaCustom
from cc3d.twedit5.PrinterTwedit import PrinterTwedit
from cc3d.twedit5.KeyboardShortcutsDlg import KeyboardShortcutsDlg
from cc3d.twedit5.DataSocketCommunicators import FileNameReceiver
from cc3d.twedit5 import Encoding, __version__, __revision__, __commit_tag__
from cc3d.twedit5.twedit.utils.collection_utils import remove_duplicates
# from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF32
from cc3d.twedit5.PluginManager.PluginManager import PluginManager
from cc3d.twedit5.ThemeManager import ThemeManager

coding_regexps = [
    (2, re.compile(r'''coding[:=]\s*([-\w_.]+)''')),
    (1, re.compile(r'''<\?xml.*\bencoding\s*=\s*['"]([-\w_.]+)['"]\?>''')),
]

from cc3d.twedit5.findandreplacedlg import ALL_IN_FILES, ALL_IN_ALL_OPEN_DOCS, ALL_IN_CURRENT_DOC
from cc3d.twedit5.Messaging import stdMsg, dbgMsg, pd, errMsg, setDebugging


class ChangedTextHandler:
    """
        ChangedTextHandler is responsible for changing editor tab icons to indicate modification of the document.
    """

    def __init__(self, _editor, _editorWindow):

        self.editor = _editor

        self.editorWindow = _editorWindow

    def handleChangedText(self):

        self.editorWindow.updateTextSizeLabel()

    def handleModificationChanged(self, m):

        """

            Slot handleModificationChanged changes editor tab icons to indicate modification of the document.

        """

        self.editorWindow.setWindowModified(m)

        current_tab_widget = self.editorWindow.panels[0]

        if self.editor.panel:
            current_tab_widget = self.editor.panel

            # for rean only editors we do not change tab icons

        if self.editor.isReadOnly(): return

        if m:  # document has been modified
            index = current_tab_widget.indexOf(self.editor)
            current_tab_widget.setTabIcon(index, QtGui.QIcon(':/icons/document-edited.png'))

        else:  # document has been restored to original state
            index = current_tab_widget.indexOf(self.editor)
            current_tab_widget.setTabIcon(index, QtGui.QIcon(':/icons/document-clean.png'))

        self.editorWindow.checkActions()


class CustomTabBar(QTabBar):
    """

        CustomTabBar is used as a component of tab widgets (CustomTabWidget) used in EditorWindow

    """

    def __init__(self, _parent=None):
        QTabBar.__init__(self, _parent)

        self.tabWidget = _parent

        self.setStyleSheet("QTabBar::tab { height: 20px;}")

        self.clickedTabPosition = -1

    def mousePressEvent(self, event):
        self.clickedTabPosition = self.tabWidget.tabBar().tabAt(event.pos())

        self.tabWidget.widget(self.clickedTabPosition).setFocus(Qt.MouseFocusReason)

        super(self.__class__, self).mousePressEvent(event)

    def contextMenuEvent(self, event):
        self.tabWidget.contextMenuEvent(event)


class CustomTabWidget(QTabWidget):
    """

        CustomTabWidget is used in place of default implementation to intercept and handle context menus when user right-clicks editor tab

    """

    def __init__(self, _parent=None):
        QTabWidget.__init__(self, _parent)

        self.editorWindow = _parent

        self.editorWindow.clickedTabWidget = None

        self.clickedTabPosition = -1

        self.tabBarLocal = CustomTabBar(self)

        self.setTabBar(self.tabBarLocal)

    def contextMenuEvent(self, event):
        """

            contextMenuEvent handles right clicks on the editor tabs

        """

        #         self.clickedTabPosition=self.tabBar().tabAt(event.pos())

        self.clickedTabPosition = self.tabBarLocal.clickedTabPosition

        self.setCurrentWidget(self.widget(self.clickedTabPosition))

        self.editorWindow.clickedTabWidget = self

        self.editorWindow.activeTabWidget = self

        menu = QMenu(self)

        menu.addAction(am.actionDict["Save"])

        menu.addAction(am.actionDict["Save As..."])

        menu.addAction(am.actionDict["Print..."])

        menu.addSeparator()

        # ---------------------------

        menu.addAction(am.actionDict["Close Tab"])

        menu.addAction(am.actionDict["Close All Tabs"])

        menu.addAction(am.actionDict["Close All But Current Tab"])

        menu.addAction(am.actionDict["Delete from disk"])

        menu.addSeparator()

        # ---------------------------

        fnToClipAct = menu.addAction("File name to clipboard")

        fdToClipAct = menu.addAction("File directory to clipboard")

        menu.addSeparator()

        # ---------------------------

        menu.addAction(am.actionDict["Move To Other View"])

        fnToClipAct.triggered.connect(self.editorWindow.fileNameToClipboard)

        fdToClipAct.triggered.connect(self.editorWindow.fileDirToClipboard)

        # self.connect(fnToClipAct, SIGNAL("triggered()"), self.editorWindow.fileNameToClipboard)

        # self.connect(fdToClipAct, SIGNAL("triggered()"), self.editorWindow.fileDirToClipboard)

        menu.exec_(event.globalPos())


class EditorWindow(QMainWindow):

    def __init__(self, _startFileListener=True):

        """

            Constructor of main window

        """
        QMainWindow.__init__(self)

        self.extensionLanguageMap = {".py": "Python",
                                     ".pyw": "Python",
                                     ".xml": "XML",
                                     ".cc3d": "XML",
                                     ".c": "C",
                                     ".cpp": "C",
                                     ".cc": "C",
                                     ".cxx": "C",
                                     ".h": "C",
                                     ".hxx": "C",
                                     ".hpp": "C",
                                     ".cmake": "CMake",
                                     ".cs": "C#",
                                     ".css": "CSS",
                                     ".d": "D",
                                     ".diff": "Diff",
                                     ".patch": "Diff",
                                     ".pas": "Pascal",
                                     ".inc": "Pascal",
                                     ".pl": "Perl",
                                     ".rb": "Ruby",
                                     ".rbw": "Ruby",
                                     ".f": "Fortran77",
                                     ".F": "Fortran77",
                                     ".f90": "Fortran",
                                     ".F90": "Fortran",
                                     ".java": "Java",
                                     ".js": "JavaScript",
                                     ".json": "JSON",
                                     ".lua": "Lua",
                                     ".m": "Matlab",
                                     ".oct": "Octave",
                                     ".pro": "IDL",
                                     ".ps": "PostScript",
                                     ".properties": "Properties",
                                     ".pov": "POV",
                                     ".cir": "Spice",
                                     ".sql": "SQL",
                                     ".tcl": "TCL",
                                     ".v": "Verilog",
                                     ".vhd": "VHDL",
                                     ".vhdl": "VHDL",
                                     ".yml": "YML",
                                     ".bat": "Batch",
                                     ".sh": "Bash",
                                     ".html": "HTML",
                                     ".tex": "TeX"

                                     }

        self.configuration = Configuration()

        self.toolbarIconSize = QSize(32, 32)

        # used to manage color themes for Twedit++5- parses xml configuration files written using notepad++ convention

        self.themeManager = ThemeManager()

        self.themeManager.readThemes()

        self.currentThemeName = str(self.configuration.setting("Theme"))

        self.fileDialogFilters, self.filterExtensionsDict = self.prepareFileDialogFilters(self.extensionLanguageMap)

        self.findDialogForm = None  # reference to findDialogForm - we check if current value is non-None to make sure we can create an instance. Multiple instances are disallowed

        self.keyboardShortcutDlg = None  # reference to keyboardShortcut dialog

        if _startFileListener:
            self.fileNameReceiver = FileNameReceiver(self)

            self.fileNameReceiver.newlyReadFileName.connect(self.processNewlyReadFileName)

        self.textSizeLabel = QLabel("SIZE LABEL")

        self.currentPositionLabel = QLabel("CURRENT POSITION")

        self.encodingLabel = QLabel("Encoding")

        self.statusBar().addPermanentWidget(self.textSizeLabel)

        self.statusBar().addPermanentWidget(self.currentPositionLabel)

        self.statusBar().addPermanentWidget(self.encodingLabel)

        self.deactivateChangeSensing = False

        self.curFile = ''

        self.zoomRange = self.configuration.setting("ZoomRange")

        self.resize(self.configuration.setting("InitialSize"))

        self.move(self.configuration.setting("InitialPosition"))

        self.baseFont = QFont(self.configuration.setting("BaseFontName"),

                              int(self.configuration.setting("BaseFontSize")))

        self.panels = []

        self.panels.append(CustomTabWidget(self))

        self.panels.append(CustomTabWidget(self))

        self.panels[0].setTabsClosable(True)

        self.panels[0].setMovable(True)

        self.panels[0].setUsesScrollButtons(True)

        # self.panels[0].tabBar().currentChanged.connect(self.__tabIndexChanged)

        for panel in self.panels:
            panel.setTabsClosable(True)

            panel.setMovable(True)

            panel.setUsesScrollButtons(True)

        self.clickedTabWidget = None  # stores a reference to recently clicked tab widget

        self.activeTabWidget = None  # stores a reference to active tab widget - the one which has focus

        self.splitter = QSplitter(self)

        self.splitter.addWidget(self.panels[0])

        self.splitter.addWidget(self.panels[1])

        self.panels[1].hide()

        self.setCentralWidget(self.splitter)

        self.defaultEditor = None

        self.textChangedHandlers = {}

        # class variables used in searches

        self.textToFind = ""

        self.indendationGuidesColor = QColor("silver")

        # bookmarks setup

        dummyEditor = QsciScintillaCustom()

        # self.lineBookmark = dummyEditor.markerDefine(QsciScintilla.SC_MARK_SHORTARROW) #All editors tab share same markers

        self.setEditorProperties(dummyEditor)

        dummyEditor.markerAdd(0, self.lineBookmark)

        self.bookmarkMask = dummyEditor.markersAtLine(0)

        dummyEditor.markerDelete(0)

        self.findAndReplaceHistory = FindAndReplaceHistory()

        self.restoreFindAndReplaceHistory(self.findAndReplaceHistory)

        self.commentStyleDict = {}

        # # # self.commentStyleDict={self.panels[0].currentWidget():['','']}

        # # # dbgMsg("textEditLocal.braceMatching()=",textEditLocal.braceMatching())

        self.createActions()

        self.createMenus()

        self.createToolBars()

        self.createStatusBar()

        self.languageManager = LanguageManager(self)

        self.languageManager.createActions()

        # setting up dock window to display Find in files results

        self.findDock = self.__createDockWindow("Find Results")

        self.findDisplayWidget = FindDisplayWidget(self)

        self.__setupDockWindow(self.findDock, Qt.BottomDockWidgetArea, self.findDisplayWidget, "Find In Files Results")

        # self.__setupDockWindow(self.findDock, Qt.BottomDockWidgetArea, self.findDisplayWidget,

        #                        self.trUtf8("Find In Files Results"))

        self.themeManager.applyThemeToEditor(self.currentThemeName, self.findDisplayWidget)

        # sys.exit()

        self.setCurrentFile('')

        self.setUnifiedTitleAndToolBarOnMac(True)

        self.fileDict = {}  # we will store information about editor, file name, modificationtime and encoding in a dictionary:

        # {key=editorwidget: [fileName,modificationTime,encoding]}

        textEditLocal = None

        if self.configuration.setting("RestoreTabsOnStartup"):
            self.restoreTabs()

        if not self.panels[0].count() and not self.panels[1].count():

            textEditLocal = QsciScintillaCustom(self, self.panels[0])

            self.panels[0].addTab(textEditLocal, QtGui.QIcon(':/icons/document-clean.png'),

                                  "New Document " + str(self.panels[0].count() + 1))

            textEditLocal.setFocus(Qt.MouseFocusReason)

            self.commentStyleDict = {self.panels[0].currentWidget(): ['', '']}

            self.textChangedHandlers[textEditLocal] = ChangedTextHandler(textEditLocal, self)

            self.panels[0].widget(0).modificationChanged.connect(

                self.textChangedHandlers[textEditLocal].handleModificationChanged)

            self.panels[0].widget(0).textChanged.connect(self.textChangedHandlers[textEditLocal].handleChangedText)

            self.panels[0].widget(0).cursorPositionChanged.connect(self.handleCursorPositionChanged)

            lexer = textEditLocal.lexer()

            if lexer:
                lexer.setFont(self.baseFont)

            textEditLocal.setFont(self.baseFont)

            self.setEditorProperties(textEditLocal)

        if not self.panels[0].count():
            self.panels[0].hide()

        self.lastFileOpenPath = ""

        self.ctrlTabShortcut = QShortcut(QKeySequence("Ctrl+Tab"), self)

        self.CtrlKeyEquivalent = Qt.Key_Control

        if sys.platform.startswith("darwin"):
            self.ctrlTabShortcut = QShortcut(QKeySequence("Alt+Tab"), self)

            self.CtrlKeyEquivalent = Qt.Key_Alt

        self.ctrlTabShortcut.activated.connect(self.cycleTabs)

        # self.connect(self.ctrlTabShortcut, SIGNAL("activated()"), self.cycleTabs)

        self.ctrlPressed = False  # used as a flag to display list of open tabs during cycling

        self.cycleTabsFlag = False

        self.cycleTabsPopup = None

        self.cycleTabFilesList = CycleTabFileList(self)

        self.pm = PluginManager(self)

        self.setDefaultStyling()

        # # # lexer=textEditLocal.lexer()

        # # # if lexer:

        # # # lexer.setFont(self.baseFont)

        # # # textEditLocal.setFont(self.baseFont)

        self.argv = None

        self.processId = 0  # used only in windows

        # def __tabIndexChanged(self,_idx):

        # print "index changed=",_idx

    def display_popup_message(self, message_title, message_text, message_type='WARNING',

                              buttons=QtWidgets.QMessageBox.Ok):

        if message_type == 'WARNING':
            ret = QtWidgets.QMessageBox.warning(self, message_title, message_text, buttons)

    def setDefaultStyling(self):

        for toolBarName, toolBar in self.toolBar.items():
            toolBar.setIconSize(self.toolbarIconSize)

            # fcns for accessing/manipulating dictionary storing tab/editor information

    def setProcessId(self, _id):

        """

            On Widnows used to set process id - used to put twedit as a top window when invoking it from external program

        """

        self.processId = _id

    def getProcessId(self):

        """

            On Widnows returns process id - used to put twedit as a top window when invoking it from external program

        """

        return self.processId

    def getEditorList(self):

        """

            returns dictionary indexed by editor widgets - {key=editorwidget: [fileName,modificationTime,encoding]}

        """

        return list(self.fileDict.keys())

    def getCurrentDocumentName(self):

        """

            returns full name of the active editor - the one which has keyboard focus

        """

        return self.getEditorFileName(self.getActiveEditor())

    def getCurrentEditor(self):

        """

            returns active editor - the one which has keyboard focus

        """

        return self.getActiveEditor()

    def getCurrentTab(self):

        """

            returns active editor tab widget - the one which has keyboard focus

        """

        return self.activeTabWidget

    def getCurrentTabWidgetAndIndex(self):

        """

            returns active tab widget and current index- the one which has keyboard focus

        """

        activePanel = self.getActivePanel()

        tabIndex = activePanel.indexOf(activePanel.currentWidget())

        return activePanel, tabIndex

    def getTabWidgetAndWidgetIndex(self, _wgt):

        tabIndex = self.panels[0].indexOf(_wgt)

        if tabIndex >= 0:

            return self.panels[0], tabIndex

        else:

            tabIndex = self.panels[1].indexOf(_wgt)

            if tabIndex >= 0:

                return self.panels[1], tabIndex

            else:

                return None, -1

    def checkIfEditorExists(self, _editor):

        """

            checks if _editor is in the self.fileDict

        """

        try:

            self.fileDict[_editor]

            return True

        except KeyError:

            return False

    def getEditorFileName(self, _editor):

        """

            returns file name for _editor. If cannot find _editor in fileDict returns ''

        """

        try:

            return self.fileDict[_editor][0]

        except (KeyError, IndexError):

            return ''

    def getEditorFileModificationTime(self, _editor):

        """

            returns file modification for _editor. If unsuccesful returns 0

        """

        try:

            return self.fileDict[_editor][1]

        except (KeyError, IndexError):

            return 0

    def getEditorFileEncoding(self, _editor):

        """

            returns file encoding for _editor. If unsuccesful returns ''

        """

        try:

            return self.fileDict[_editor][2]

        except (KeyError, IndexError):

            return ''

    def removeEditor(self, _editor):

        """

            removes entry associated with  _editor from fileDict

        """

        try:

            del self.fileDict[_editor]

        except KeyError:

            pass

    def setEditorFileName(self, _editor, _name):

        """

            sets file name for  _editor in fileDict

        """

        try:

            self.fileDict[_editor][0] = _name

        except (KeyError, IndexError):

            pass

    def setEditorFileModificationTime(self, _editor, _time):

        """

            sets file modification time for  _editor in fileDict

        """

        try:

            self.fileDict[_editor][1] = _time

        except (KeyError, IndexError):

            pass

    def setEditorFileEncoding(self, _editor, _encoding):

        """

            sets file encoding for  _editor in fileDict

        """

        try:

            self.fileDict[_editor][2] = _encoding

        except (KeyError, IndexError):

            pass

    def setPropertiesInEditorList(self, _editor, _fileName='', _modificationTime=0, _encoding=''):

        """

            sets file name, file modification time and file encoding for  _editor in fileDict

        """

        try:

            self.fileDict[_editor][0] = _fileName

            self.fileDict[_editor][1] = _modificationTime

            self.fileDict[_editor][2] = _encoding

        except KeyError:

            self.fileDict[_editor] = [_fileName, _modificationTime, _encoding]

        except IndexError:

            pass

    def getPropertiesFromEditorList(self, _editor):

        """

            returns file name, file modification time and file encoding for  _editor from file Dict.

            If unsuccesfull it returns ('',0,'')

        """

        try:

            return self.fileDict[_editor]

        except KeyError:

            return ('', 0, '')

    # fcns dealing with status bar

    def handleCursorPositionChanged(self, _line, _index):

        """

            slot handling change in the cursor position - updates status bar

        """

        self.currentPositionLabel.setText("Ln : %s  Col : %s" % (_line, _index))

    def updateTextSizeLabel(self):

        """

            slot handling change in text size (measured in the number of characters) - updates status bar

        """

        editor = self.getActiveEditor()

        try:  # in case editor is None

            self.textSizeLabel.setText("Length : %s  lines : %s" % (editor.length(), editor.lines()))

        except:

            pass

    def updateEncodingLabel(self):

        """

            slot handling file encoding - updates status bar

        """

        editor = self.getActiveEditor()

        try:

            self.encodingLabel.setText(Encoding.normalizeEncodingName(self.getEditorFileEncoding(editor)))



        except KeyError:

            self.encodingLabel.setText("Unknown encoding")

    def prepareFileDialogFilters(self, _extensionLanguageMap):

        """

            Helper function used to initialze pull-down list of file types for file dialogs (open, sae, save as etc)

        """

        filterList = ''

        filterDict = {}

        for extension, language in _extensionLanguageMap.items():

            dbgMsg("extension=", extension, " language=", language)

            try:

                filterDict[language] = filterDict[language] + " *" + extension

            except KeyError:

                filterDict[language] = "*" + extension

        keysSorted = list(filterDict.keys())

        keysSorted.sort()

        # filterDict={} # reassign filter dict

        for language in keysSorted:

            if language == "C":
                # filterList.append("C/C++"+" file ("+filterDict[language]+");;")

                filterList += "C/C++" + " file (" + filterDict[language] + ");;"

                continue

            if language == "CMake":
                # filterList.append(language+" file ("+filterDict[language]+" CMakeLists.*);;")

                filterList += language + " file (" + filterDict[language] + " CMakeLists.*);;"

                continue

            # filterList.append(language+" file ("+filterDict[language]+");;")

            filterList += language + " file (" + filterDict[language] + ");;"

            # filterList.insert(0,"Text file (*.txt);;")

        # filterList.insert(0,"All files (*);;")

        filterList = "Text file (*.txt);;" + filterList

        filterList = "All files (*);;" + filterList

        return filterList, filterDict

    # def prepareFileDialogFilters(self, _extensionLanguageMap):

    #     """

    #         Helper function used to initialze pull-down list of file types for file dialogs (open, sae, save as etc)

    #     """

    #     filterList = ''

    #     filterDict = {}

    #     for extension, language in _extensionLanguageMap.iteritems():

    #         dbgMsg("extension=", extension, " language=", language)

    #         try:

    #             filterDict[language] = filterDict[language] + " *" + extension

    #         except KeyError:

    #             filterDict[language] = "*" + extension

    #

    #     keysSorted = filterDict.keys()

    #     keysSorted.sort()

    #

    #     # filterDict={} # reassign filter dict

    #

    #     for language in keysSorted:

    #         if language == "C":

    #             filterList.append("C/C++" + " file (" + filterDict[language] + ");;")

    #             continue

    #         if language == "CMake":

    #             filterList.append(language + " file (" + filterDict[language] + " CMakeLists.*);;")

    #             continue

    #

    #         filterList.append(language + " file (" + filterDict[language] + ");;")

    #

    #     filterList.insert(0, "Text file (*.txt);;")

    #     filterList.insert(0, "All files (*);;")

    #     return filterList, filterDict

    def getCurrentFilterString(self, _extension):

        """

            Helper function  - returns filter based on file extension

        """

        currentFilter = ""

        language = ""

        if _extension == ".txt":
            return "Text file (*.txt)"

        try:

            language = self.extensionLanguageMap[_extension]



        except KeyError:

            return currentFilter

        try:

            if language == "C":

                currentFilter = "C/C++" + " file (" + self.filterExtensionsDict[language] + ")"

            else:

                currentFilter = language + " file (" + self.filterExtensionsDict[language] + ")"

            return currentFilter

        except KeyError:

            return currentFilter

    def getFileNameToEditorWidgetMap(self):

        """

            translates file dict to provide dictionary indexed {fileName:editor} - it is in essence reverse dictionary of fileDict

        """

        openFileDict = {}

        # key is QScintilla widget

        for key in list(self.fileDict.keys()):

            if self.fileDict[key][0] != '':

                openFileDict[self.fileDict[key][0]] = key

            else:

                documentName = key.panel.tabText(key.panel.indexOf(key))

                if documentName != '':
                    openFileDict[documentName] = key

        return openFileDict

    def get_editor_file_name(self, editor):
        """
        Return file name given editor as an argument. Returns None if fname cannot be found
        :param editor:
        :return:
        """

        try:
            return self.fileDict[editor][0]
        except (KeyError, IndexError):
            pass
        # open_file_dict = self.getFileNameToEditorWidgetMap()
        #
        # for f_name, editor_local in open_file_dict.items():
        #     if editor_local == editor:
        #         return f_name

    def getActiveEditor(self):

        """

            returns editor with current keyboard focus - active editor

        """

        if self.activeTabWidget:

            return self.activeTabWidget.currentWidget()

        else:  # default choice when active tab widget is None

            return self.panels[0].currentWidget()

    def setActiveEditor(self, _editor):

        ''' sets _editor widget active - chooses correct panel and then makes _editor active '''

        # locating index of the widget and a pane to which it belongs

        widgetIndex = -1

        panelIndex = -1

        for i in range(2):

            widgetIndex = self.panels[i].indexOf(_editor)

            if widgetIndex >= 0:
                panelIndex = i

                break

        if widgetIndex >= 0 and panelIndex >= 0:
            self.panels[panelIndex].setCurrentWidget(_editor)

            _editor.setFocus(Qt.MouseFocusReason)

    def getActivePanel(self):

        """

            returns panel to which active editor belongs. If active editor is None it returns first panel

        """

        if self.getActiveEditor():

            return self.getActiveEditor().panel

        else:

            return self.panels[0]

    def cycleTabs(self):

        """

            displays widget with file names of all open editors - invoked when user presses ctrl+Tab (or alt+Tab on Mac). It cycles through open editors

        """

        dbgMsg("CYCLE TABS")

        if not self.cycleTabsFlag:  # first time entering cycle tabs - prepare list of files

            self.cycleTabsFlag = True

            dbgMsg("Preparing list of Tabs")

            self.cycleTabsPopup = CycleTabsPopup(self)

            self.cycleTabFilesList.initializeTabFileList()

        if self.cycleTabFilesList:
            self.cycleTabFilesList.initializeTabFileList()

            self.cycleTabsPopup.initializeContent(self.cycleTabFilesList)

            self.cycleTabsPopup.move(0, 0)  # always position popup at (0,0) then calculate required shift

            self.cycleTabsPopup.adjustSize()

            # setting position of the popup cycle tabs window

            geom = self.geometry()

            pGeom = self.cycleTabsPopup.geometry()

            pCentered_x = geom.x() + (geom.width() - pGeom.width()) / 2

            pCentered_y = geom.y() + (geom.height() - pGeom.height()) / 2

            self.cycleTabsPopup.move(pCentered_x, pCentered_y)

            pGeom = self.cycleTabsPopup.geometry()

            self.cycleTabsPopup.show()

            dbgMsg("self.cycleTabsPopup.pos()=", str(self.cycleTabsPopup.pos()) + "\n\n\n\n")

    def keyPressEvent(self, event):

        """

            senses if Ctrl was pressed by user

        """

        if event.key() == self.CtrlKeyEquivalent:
            dbgMsg("CTRL key pressed")

            self.ctrlPressed = True

    def keyReleaseEvent(self, event):

        """

            Senses if Ctrl was released - if so it closes cycleTab popup widget

        """

        if event.key() == self.CtrlKeyEquivalent:

            dbgMsg("CTRL RELEASED")

            self.ctrlPressed = False

            if self.cycleTabsFlag:  # release cycleTabs flag and switch to new tab

                self.cycleTabsFlag = False

                self.cycleTabsPopup.close()

                self.cycleTabsPopup = None

    def onMarginClick(self, _margin, _line, _state):

        dbgMsg("_margin:", _margin, " line:", _line, " _state:", _state)

    def openFileList(self, _fileList):

        """

            opens list of files (stored in _fileList)

        """

        for file_name in _fileList:
            dbgMsg("THIS IS THE FILE TO BE OPEN:", os.path.abspath(str(file_name)))

            self.loadFile(os.path.abspath(str(file_name)))

        self.activateWindow()

    def processCommandLine(self):
        """
            opens file names listed as cmd line arguments
        """

        dbgMsg("\n\n\n\n\n PROCESSING CMD LINE")

        for file_name in self.argv:
            dbgMsg("THIS IS THE FILE TO BE OPEN:", os.path.abspath(str(file_name)))

            self.loadFile(os.path.abspath(str(file_name)))

        self.activateWindow()

    def processNewlyReadFileName(self, _fileName):

        """

            loads new file

        """

        dbgMsg("processNewlyReadFileName fileName=", _fileName)

        self.loadFile(os.path.abspath(str(_fileName)))

        if not sys.platform.startswith('win'):
            self.showNormal()

            self.activateWindow()

            self.raise_()

            self.setFocus(Qt.MouseFocusReason)

            # if sys.platform!='win32':

            # def mousePressEvent(self,event):

            # print "mousePressEvent EditorWindow"

    def wheelEvent(self, event):

        """

            handles mouse wheel event if ctrl is pressed it zooms in or out depending on wheel movement direction

        """

        dbgMsg("WHEEL EVENT")

        if qApp.keyboardModifiers() == Qt.ControlModifier:
            y_scroll_delta = event.angleDelta()

            if isinstance(y_scroll_delta, QPoint):
                y_scroll_delta = y_scroll_delta.y()

            if y_scroll_delta > 0:

                self.zoomIn()

            else:

                self.zoomOut()

            dbgMsg("WHEEL EVENT WITH CTRL")

    def restoreTabs(self):

        """

            restores tabs based on the last saved session

        """

        fileList = self.configuration.setting("ListOfOpenFilesAndPanels")

        for i in range(0, len(fileList), 2):
            panel = int(fileList[i + 1])

            self.loadFile(fileList[i], True, panel)

        self.splitter.restoreState(self.configuration.setting("PanelSplitterState"))

        currentTabIndex = self.configuration.setting("CurrentTabIndex")

        currentPanelIndex = self.configuration.setting("CurrentPanelIndex")

        # print "currentTabIndex=",currentTabIndex," currentPanelIndex=",currentPanelIndex

        self.panels[currentPanelIndex].setCurrentIndex(currentTabIndex)

        # self.panels[currentPanelIndex].widget(currentTabIndex).setFocus(Qt.MouseFocusReason)

    def setArgv(self, _argv):

        """

            used to store cmd line args in self.argv

        """

        dbgMsg("\n\n\n command line arguments", _argv)

        self.argv = _argv

    def getArgv(self):

        """

            returns cmd line args stored in self.argv

        """

        return self.argv

    def closeEvent(self, event):

        """

            called when user closes Main window. This fcn records current state of editor . It also unloads plugins

        """

        # todo 3 - reenable it
        # self.pm.unloadPlugins()

        # openFilesToRestore={}

        openFilesToRestore = [{}, {}]

        self.deactivateChangeSensing = True

        # determining index of the current tab

        activePanel = self.getActivePanel()

        activePanelIdx = 0

        if activePanel == self.panels[1]:
            activePanelIdx = 1

        activeEditor = self.getActiveEditor()

        activeEditorIdx = activePanel.indexOf(activeEditor)

        self.configuration.setSetting("CurrentTabIndex", activeEditorIdx)

        self.configuration.setSetting("CurrentPanelIndex", activePanelIdx)

        # print "activeEditorIdx=",activeEditorIdx," activePanelIdx=",activePanelIdx

        # only file names which are assigned prior to closing are being stored

        dbgMsg("\n\n\n self.getEditorList()=", self.getEditorList())

        # dbgMsg("self.fileDict=",self.fileDict)

        for editor in self.getEditorList():

            dbgMsg("OPEN FILE NAME=", self.getEditorFileName(editor))

            if not self.getEditorFileName(editor) == '':

                if editor.panel == self.panels[0]:

                    openFilesToRestore[0][self.panels[0].indexOf(editor)] = self.getEditorFileName(

                        editor)  # saving file name and position of the tab

                else:

                    openFilesToRestore[1][self.panels[1].indexOf(editor)] = self.getEditorFileName(

                        editor)  # saving file name and position of the tab

            # print "textEditLocal=",textEditLocal

            editor.setFocus(Qt.MouseFocusReason)

            maybeSaveFlag = self.maybeSave(editor)

            if maybeSaveFlag:

                if editor.panel == self.panels[0]:

                    openFilesToRestore[0][self.panels[0].indexOf(editor)] = self.getEditorFileName(

                        editor)  # saving file name and position of the tab

                else:

                    openFilesToRestore[1][self.panels[1].indexOf(editor)] = self.getEditorFileName(

                        editor)  # saving file name and position of the tab

        dbgMsg(" \n\n\n\n openFilesToRestore=", openFilesToRestore)

        self.deactivateChangeSensing = False

        # saving open file names to restore

        self.saveSettingsOpenFilesToRestore(openFilesToRestore)

        # saving find and replace history

        self.saveSettingsFindAndReplaceHistory(self.findAndReplaceHistory)

        self.configuration.setSetting("InitialSize", self.size())

        # self.configuration.setSetting("InitialState",self.saveState()) #  state of docking windows and their position

        self.configuration.setSetting("InitialPosition", self.pos())

        self.configuration.setSetting("PanelSplitterState", self.splitter.saveState())

        # prepare to store reassgined shortcuts

        self.configuration.prepareKeyboardShortcutsForStorage()

        self.configuration.preparePluginAutoloadDataForStorage()

        event.accept()

    def saveSettingsOpenFilesToRestore(self, openFilesToRestore):

        """

            called from closeEvent to save names of open files

        """

        keys = list(openFilesToRestore[0].keys())

        keys.sort()  # sorting by the tab number

        fileNamesSorted = [openFilesToRestore[0][key] for key in keys]

        fileList = []

        dbgMsg(dir(fileList))

        for fileName in fileNamesSorted:
            fileList.append(fileName)

            fileList.append(str(0))

        keys = list(openFilesToRestore[1].keys())

        keys.sort()  # sorting by the tab number

        fileNamesSorted = [openFilesToRestore[1][key] for key in keys]

        # fileList=QStringList()

        dbgMsg(dir(fileList))

        for fileName in fileNamesSorted:
            fileList.append(fileName)

            fileList.append(str(1))

        dbgMsg("saving len(fileList)=", len(fileList))

        # print "saving fileList.count()=",fileList.count()

        self.configuration.setSetting("ListOfOpenFilesAndPanels", fileList)

        # print "fileList=",fileList

        dbgMsg("saveSettingsOpenFilesToRestore")

    def restoreFindAndReplaceHistory(self, _frh):

        """

            restores find and replace history

        """

        findHistoryList = self.configuration.setting("FRFindHistory")

        for i in range(len(findHistoryList)):

            if str(findHistoryList[i]).strip() != '':  # we do not wantany empyty strings here

                _frh.findHistory.append(findHistoryList[i])

        replaceHistoryList = self.configuration.setting("FRReplaceHistory")

        for i in range(len(replaceHistoryList)):

            if str(replaceHistoryList[i]).strip() != '':  # we do not wantany empyty strings here

                _frh.replaceHistory.append(replaceHistoryList[i])

        filtersHistoryList = self.configuration.setting("FRFiltersHistory")

        for i in range(len(filtersHistoryList)):

            if str(filtersHistoryList[i]).strip() != '':  # we do not wantany empyty strings here

                _frh.filtersHistoryIF.append(filtersHistoryList[i])

        directoryHistoryList = self.configuration.setting("FRDirectoryHistory")

        for i in range(len(directoryHistoryList)):

            if str(directoryHistoryList[i]).strip() != '':  # we do not wantany empyty strings here

                _frh.directoryHistoryIF.append(directoryHistoryList[i])

        _frh.syntaxIndex = self.configuration.setting("FRSyntaxIndex")

        _frh.inSelection = self.configuration.setting("FRInSelection")

        _frh.inAllSubfolders = self.configuration.setting("FRInAllSubfolders")

        _frh.opacity = self.configuration.setting("FROpacity")

        _frh.transparencyEnable = self.configuration.setting("FRTransparencyEnable")

        _frh.opacityOnLosingFocus = self.configuration.setting("FROnLosingFocus")

        _frh.opacityAlways = self.configuration.setting("FRAlways")

    def saveSettingsFindAndReplaceHistory(self, _frh):

        """

            called from closeEvent to save find and replace history

        """

        findHistoryList = []

        for findText in _frh.findHistory:
            findHistoryList.append(findText)

        self.configuration.setSetting("FRFindHistory", findHistoryList)

        replaceHistoryList = []

        for replaceText in _frh.replaceHistory:
            replaceHistoryList.append(replaceText)

        self.configuration.setSetting("FRReplaceHistory", replaceHistoryList)

        filtersHistoryList = []

        for filtersText in _frh.filtersHistoryIF:
            filtersHistoryList.append(filtersText)

        self.configuration.setSetting("FRFiltersHistory", filtersHistoryList)

        directoryHistoryList = []

        for directoryText in _frh.directoryHistoryIF:
            directoryHistoryList.append(directoryText)

        self.configuration.setSetting("FRDirectoryHistory", directoryHistoryList)

        self.configuration.setSetting("FRSyntaxIndex", _frh.syntaxIndex)

        self.configuration.setSetting("FRInSelection", _frh.inSelection)

        self.configuration.setSetting("FRInAllSubfolders", _frh.inAllSubfolders)

        self.configuration.setSetting("FROpacity", _frh.opacity)

        self.configuration.setSetting("FRTransparencyEnable", _frh.transparencyEnable)

        self.configuration.setSetting("FROnLosingFocus", _frh.opacityOnLosingFocus)

        self.configuration.setSetting("FRAlways", _frh.opacityAlways)

        dbgMsg("saveSettingsFindAndReplaceHistory")

    # QChangeEvent detects changes in application status such as e.g. being activated or not see qt docuentation for more

    # see also QEvent documentation for list of type of events

    # event.type() prints numeric value of gien event

    def changeEvent(self, event):

        """

            detects changes in application status such as e.g. being activated. used to figure out if document was modified

        """

        if self.deactivateChangeSensing:
            return

        dbgMsg("THIS IS CHANGE EVENT ", event.type())

        if event.type() == QEvent.ActivationChange:

            dbgMsg("application focus has changed")

            dbgMsg("isActiveWindow()=", self.isActiveWindow())

            if self.isActiveWindow():
                self.deactivateChangeSensing = True

                self.checkIfDocumentsWereModified()

                self.deactivateChangeSensing = False

    def checkIfDocumentsWereModified(self):

        """

            checks if document has been modified

        """

        self.deactivateChangeSensing = True

        for editor in self.getEditorList():

            fileName = self.getEditorFileName(editor)

            try:

                if fileName != '' and os.path.getmtime(str(fileName)) != self.getEditorFileModificationTime(editor):

                    dbgMsg("DOCUMENT ", fileName, " was modified")

                    reload_flag = self.maybeReload(editor)

                    if not reload_flag:
                        self.setEditorFileModificationTime(editor, os.path.getmtime(str(fileName)))

                    dbgMsg("reloadFlag=", reload_flag)

                    # check if the document exists at all

            except os.error:

                if not editor.isModified():

                    message = "File <b>\"%s\"</b>  <br> has been deleted <br> Keep it in Editor?" % fileName

                    ret = QtWidgets.QMessageBox.warning(self, "Missing File", message,

                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

                    if ret == QtWidgets.QMessageBox.Yes:

                        lastModTime = self.getEditorFileModificationTime(editor)

                        self.deactivateChangeSensing = False

                        # here we will insert and remove space at the end of the line - enough to trigger modification signal of the document

                        self.setEditorFileModificationTime(editor, lastModTime + 1)

                        lineNo = editor.lines() - 1  # lines are numbnered from 0

                        lineLength = editor.lineLength(lineNo)

                        editor.insertAt(' ', lineNo, lineLength)

                        editor.setSelection(lineNo, lineLength, lineNo, lineLength + 1)

                        editor.removeSelectedText()

                        self.deactivateChangeSensing = True

                    else:

                        self.closeTab(editor.panel.indexOf(editor), False)

        self.deactivateChangeSensing = False

    def setEditorProperties(self, _editor):

        """

            for each newly created editor this fcn sets all editor properties based on twedit configuration e.g. font size, display of EOL etc

        """

        # this is essential on OSX otherwise after changing lexer line number font changes

        _editor.setMarginsFont(self.baseFont)  # we first set font for margin

        # lines are displayed on margin 0

        lineNumbersFlag = self.configuration.setting("DisplayLineNumbers")

        #         self.adjustLineNumbers(_editor,lineNumbersFlag,False) # we do not need to call it - line number adjuster is called automatically in QsciScintillaCustom.py - self.linesChanged.connect(self.linesChangedHandler)

        useTabSpaces = self.configuration.setting("UseTabSpaces")

        _editor.setIndentationsUseTabs(not useTabSpaces)

        if useTabSpaces:

            _editor.setIndentationWidth(self.configuration.setting("TabSpaces"))

        else:

            _editor.setIndentationWidth(

                0)  # If width is 0 then the value returned by tabWidth() is used

        self.lineBookmark = _editor.markerDefine(QsciScintilla.SC_MARK_SHORTARROW,

                                                 0)  # All editors tab share same markers notice that ) denotes marker number - multiple calls to markerDefine with empty second argument cause marker assignment to have difference numbers and this causes problems

        # _editor.setMarginMarkerMask(2,QsciScintilla.SC_MARK_SHORTARROW) # mask has to correspond to marker number returned by markerDefine or it has to simply be 0 not sure

        # Bookmarks are displayed on margin 1 but I have to pass 2 to set mask for margin 1

        _editor.setMarginMarkerMask(2,

                                    QsciScintilla.SC_MARK_SHORTARROW)  # mask has to correspond to marker number returned by markerDefine or it has to simply be 0 not sure

        # dbgMsg("MASK 0=",_editor.marginMarkerMask(0))

        # dbgMsg("MASK 1=",_editor.marginMarkerMask(1))

        # dbgMsg("MASK 2=",_editor.marginMarkerMask(2))

        # enable bookmarking by click

        _editor.setMarginSensitivity(1, True)

        try:

            _editor.marginClicked.disconnect(self.marginClickedHandler)

        except:  # when margin clicked is disconnected call to disconnect throws exception

            pass

        _editor.marginClicked.connect(self.marginClickedHandler)

        _editor.setMarkerBackgroundColor(QColor("lightsteelblue"), self.lineBookmark)

        if self.configuration.setting("FoldText"):

            _editor.setFolding(QsciScintilla.BoxedTreeFoldStyle)

        else:

            _editor.setFolding(QsciScintilla.NoFoldStyle)

        _editor.setCaretLineVisible(True)

        _editor.setCaretLineBackgroundColor(QtGui.QColor('#EFEFFB'))

        # _editor.modificationChanged.connect(self.modificationChangedSlot)

        if not sys.platform.startswith('win'):

            _editor.setEolMode(QsciScintilla.EolUnix)

        else:

            _editor.setEolMode(QsciScintilla.EolWindows)  # windows eol only on system whose name starts with 'win'

        # _editor.setEolMode(QsciScintilla.EolWindows) # SETTING EOL TO WINDOWS MESSES THINGS UP AS SAVE FCN (MOST LIKELY)ADS EXTRA CR SIGNS -

        # _editor.setEolMode(QsciScintilla.EolUnix) # SETTING EOL TO WINDOWS MESSES THINGS UP AS SAVE FCN (MOST LIKELY)ADS EXTRA CR SIGNS -

        # _editor.setEolMode(QsciScintilla.EolMac) # SETTING EOL TO WINDOWS MESSES THINGS UP AS SAVE FCN (MOST LIKELY)ADS EXTRA CR SIGNS -

        # _editor.setUtf8(True) # using UTF-8 encoding

        displayWhitespace = self.configuration.setting("DisplayWhitespace")

        if displayWhitespace:

            _editor.setWhitespaceVisibility(QsciScintilla.SCWS_VISIBLEALWAYS)

        else:

            _editor.setWhitespaceVisibility(QsciScintilla.SCWS_INVISIBLE)

        displayEol = self.configuration.setting("DisplayEOL")

        _editor.setEolVisibility(displayEol)

        wrapLines = self.configuration.setting("WrapLines")

        showWrapSymbol = self.configuration.setting("ShowWrapSymbol")

        _editor.setAutoIndent(True)

        if wrapLines:

            if sys.platform.startswith("darwin"):

                # when opening file on Mac we simply ignore fold setting as they cause large files to open very slowly, users can set wrap lines setting individually for each editor tab

                _editor.setWrapMode(QsciScintilla.WrapNone)

            else:

                _editor.setWrapMode(QsciScintilla.WrapWord)

                if showWrapSymbol:
                    _editor.setWrapVisualFlags(QsciScintilla.WrapFlagByText)

        # Autocompletion

        if self.configuration.setting("EnableAutocompletion"):

            _editor.setAutoCompletionThreshold(self.configuration.setting("AutocompletionThreshold"))

            # _editor.setAutoCompletionSource(QsciScintilla.AcsDocument)

            _editor.setAutoCompletionSource(QsciScintilla.AcsAll)

        else:

            _editor.setAutoCompletionSource(QsciScintilla.AcsNone)

        lexer = _editor.lexer()

        if lexer:
            lexer.setFont(self.baseFont)

        _editor.setFont(self.baseFont)

        # SCI_STYLESETFORE(int styleNumber, int colour)

        # _editor.SendScintilla(QsciScintilla.SCI_STYLESETFORE,1,255)

        self.themeManager.applyThemeToEditor(self.currentThemeName, _editor)

        # self.themeManager.applyThemeToEditor('Choco',_editor)

        _editor.zoomTo(self.zoomRange)

    def modificationChangedSlot(self, _flag):

        dbgMsg("THIS IS CHANGED DOCUMENT")

    def block_comment(self):

        """

            slot called when block-commenting code

        """

        editor = self.getActiveEditor()

        if not self.commentStyleDict[editor][0]:
            # this language does not allow comments
            return

        comment_string_begin = self.commentStyleDict[editor][0]

        comment_string_end = self.commentStyleDict[editor][1]

        if editor.hasSelectedText():

            line_from, index_from, line_to, index_to = editor.getSelection()

            if index_to == 0:
                # do not comment last line if no character in this line is selected
                line_to -= 1

            num_existing_comments = self.check_for_existing_comments(
                line_from=line_from, line_to=line_to, comment_string_begin=comment_string_begin)
            if num_existing_comments == line_to - line_from + 1:
                # if every line has comments then comment action turns into uncomment
                self.block_uncomment()
                return

            editor.beginUndoAction()

            for line_no in range(line_from, line_to + 1):
                self.comment_single_line(line_no, comment_string_begin, comment_string_end)

            editor.endUndoAction()

            # restoring selection
            if index_to == 0:
                line_to += 1
                editor.setSelection(line_from, index_from + len(comment_string_begin), line_to, index_to)
            else:
                editor.setSelection(line_from, index_from + len(comment_string_begin), line_to,
                                    index_to + len(comment_string_begin))

        else:

            # single line comments
            current_line, current_column = editor.getCursorPosition()

            num_existing_comments = self.check_for_existing_comments(
                line_from=current_line, line_to=current_line, comment_string_begin=comment_string_begin)
            if num_existing_comments == 1:
                comment_string_begin, comment_string_begin_trunc = self.get_comment_string_begin(editor=editor)
                comment_string_end, comment_string_end_trunc = self.get_comment_string_end(editor=editor)

                self.uncomment_line(current_line, comment_string_begin, comment_string_begin_trunc, comment_string_end,
                                    comment_string_end_trunc)
                return

            editor.beginUndoAction()

            self.comment_single_line(current_line, comment_string_begin, comment_string_end)

            editor.endUndoAction()

    @staticmethod
    def check_if_eol_comment_already_exists(line_text, comment_string_end):
        """
        Checks if a given line_text contains closing comment
        :param line_text:
        :param comment_string_end:
        :return:
        """
        return line_text.rstrip().endswith(comment_string_end.strip())

    @staticmethod
    def check_if_multiple_begin_comments_already_exists(line_text, comment_string_begin):
        """
        Checks if a given line_text contains at least two opening comment sequence
        :param line_text:
        :param comment_string_begin:
        :return:
        """
        comment_string_begin = comment_string_begin.strip()
        pattern = f'^(({comment_string_begin})(\s*)({comment_string_begin}))'
        return re.search(pattern, line_text.lstrip())
        # lstrip_line_text = line_text.lstrip()
        # first_comment_pos = lstrip_line_text.find(comment_string_begin.strip()):
        #
        # return .endswith(comment_string_end.strip())

    @staticmethod
    def remove_line(editor, line_num):
        line_len = editor.lineLength(line_num)
        editor.setSelection(line_num, 0, line_num, line_len-1)
        editor.removeSelectedText()

    def lreplace(self, pattern, sub, string):
        """
        Replaces 'pattern' in 'string' with 'sub' if 'pattern' starts 'string'.
        """
        first_pos_non_whitespace = self.first_non_whitespace_pos(string)
            # len(string) - len(string.lstrip())
        openning_spaces = string[:first_pos_non_whitespace]

        replacement = re.sub('^%s' % pattern, sub, string[first_pos_non_whitespace:])
        replacement = openning_spaces + replacement

        return replacement

    def rreplace(self, pattern, sub, string):
        """
        Replaces 'pattern' in 'string' with 'sub' if 'pattern' ends 'string'.
        """
        first_end_whitespace = self.first_end_whitespace(string)
        trailing_spaces = string[first_end_whitespace:]
        replacement = re.sub('%s$' % pattern, sub, string.rstrip()) + trailing_spaces

        return replacement

    @staticmethod
    def first_non_whitespace_pos(text):
        return len(text) - len(text.lstrip())

    @staticmethod
    def first_end_whitespace(text):
        return len(text.rstrip())

    def comment_xml_begin(self, editor, cur_line_num):
        comment_begin = '<!-- '
        nested_comment_extra = '&lt;!&ndash;'
        cur_line_text = editor.text(cur_line_num)
        comment_found = self.find_comment_pos_begin(text=cur_line_text, comment=comment_begin)
        if comment_found is not None:
            cur_line_text = self.lreplace(comment_found, f'{comment_begin}{nested_comment_extra} ', cur_line_text)
            self.remove_line(editor=editor, line_num=cur_line_num)
            editor.insertAt(cur_line_text, cur_line_num, 0)
        else:
            first_pos_non_whitespace = len(cur_line_text) - len(cur_line_text.lstrip())
            editor.insertAt(comment_begin, cur_line_num, first_pos_non_whitespace)

    def find_comment_pos_end(self, text, comment):
        comments = [comment, comment.strip()]
        for comment in comments:
            if text.rstrip().endswith(comment):
                return comment
        return None

    def find_comment_pos_begin(self, text, comment):
        comments = [comment, comment.strip()]
        for comment in comments:
            if text.lstrip().startswith(comment):
                return comment
        return None

    def comment_xml_end(self, editor, cur_line_num):
        comment_end = ' -->'
        nested_comment_extra = '&ndash;&gt;'
        cur_line_text = editor.text(cur_line_num)

        comment_found = self.find_comment_pos_end(text=cur_line_text, comment=comment_end)

        if comment_found is not None:
            cur_line_text = self.rreplace(comment_found, f' {nested_comment_extra} {comment_end}', cur_line_text)

            self.remove_line(editor=editor, line_num=cur_line_num)
            editor.insertAt(cur_line_text, cur_line_num, 0)
        else:

            eol_pos = len(cur_line_text)
            if cur_line_text[eol_pos - 2] == "\r" or cur_line_text[eol_pos - 2] == "\n":
                # second option is just in case - checking if we are dealing
                # with CR LF or simple CR or LF end of line

                editor.insertAt(comment_end, cur_line_num, eol_pos - 2)

            else:

                editor.insertAt(comment_end, cur_line_num, eol_pos - 1)

    def comment_single_line(self, currentLine, commentStringBegin, commentStringEnd):

        """

            helper function called during block-commenting of the code to comment single line of code

        """

        editor = self.getActiveEditor()

        if commentStringEnd:
            # handling comments which require additions at the beginning and at the end of the line

            if editor.text(currentLine).strip():  # checking if the line contains non-white characters

                # editor.beginUndoAction()
                cur_line_text = editor.text(currentLine)
                if self.is_xml_comment(comment=commentStringBegin):
                    self.comment_xml_begin(editor=editor, cur_line_num=currentLine)
                else:
                    first_pos_non_whitespace = len(cur_line_text) - len(cur_line_text.lstrip())
                    editor.insertAt(commentStringBegin, currentLine, first_pos_non_whitespace)

                # we have to account for the fact the EOL character can be CR LF or CR or LF

                eol_pos = len(editor.text(currentLine))

                line_text = editor.text(currentLine)

                if self.is_xml_comment(comment=commentStringEnd):
                    self.comment_xml_end(editor=editor, cur_line_num=currentLine)
                else:
                    if not self.check_if_eol_comment_already_exists(
                            line_text=line_text, comment_string_end=commentStringEnd):

                        if line_text[eol_pos - 2] == "\r" or line_text[eol_pos - 2] == "\n":
                            # second option is just in case - checking if we are dealing
                            # with CR LF or simple CR or LF end of line

                            editor.insertAt(commentStringEnd, currentLine, eol_pos - 2)

                        else:

                            editor.insertAt(commentStringEnd, currentLine, eol_pos - 1)

                            # editor.endUndoAction()

        else:
            # handling comments which require additions only at the beginning of the line
            # if not editor.text(currentLine).trimmed().isEmpty():  # checking if the line contains non-white characters

            cur_line_text = editor.text(currentLine)
            if cur_line_text.strip():  # checking if the line contains non-white characters

                # editor.beginUndoAction()
                first_pos_non_whitespace = len(cur_line_text) - len(cur_line_text.lstrip())
                editor.insertAt(commentStringBegin, currentLine, first_pos_non_whitespace)

                # editor.endUndoAction()

    def copy(self):

        """

            slot - copies highlighted text to clipboard

        """

        editor = self.getActiveEditor()

        editor.copy()

    def cut(self):

        """

            slot - cuts highlighted text to clipboard

        """

        editor = self.getActiveEditor()

        editor.cut()

        # it is better to keep extended margin after removing text and wait till user saves file to adjust the width

        # if self.configuration.setting('DisplayLineNumbers') and editor.marginWidth(0):

        # self.adjustLineNumbers(editor,True)

    def paste(self):

        """

            slot - pastes  clipboard content to active editor

        """

        editor = self.getActiveEditor()

        editor.paste()

        # if self.configuration.setting('DisplayLineNumbers') and editor.marginWidth(0):

        # self.adjustLineNumbers(editor,True)

    def increaseIndent(self):

        """

            slot - increases indent of block of highlighted text

        """

        editor = self.getActiveEditor()

        line, index = editor.getCursorPosition()

        if editor.hasSelectedText():

            line_from, index_from, line_to, index_to = editor.getSelection()

            if index_to == 0:
                line_to -= 1

            editor.beginUndoAction()

            for line in range(line_from, line_to + 1):
                editor.indent(line)

            editor.endUndoAction()

        else:  # here I sohuld insert indenttion inside the string in the current position

            tabString = ""

            indentationWidth = 1

            if editor.indentationsUseTabs():

                tabString = "\t"

            else:

                indentationWidth = editor.indentationWidth()

                tabString = " " * indentationWidth

            editor.insertAt(tabString, line, index)

            editor.setCursorPosition(line, index + indentationWidth)

            # editor.indent(line)

    def decreaseIndent(self):

        """

            slot - decreases indent of block of highlighted text

        """

        editor = self.getActiveEditor()

        line, index = editor.getCursorPosition()

        if editor.hasSelectedText():

            line_from, index_from, line_to, index_to = editor.getSelection()

            if index_to == 0:
                line_to -= 1

            editor.beginUndoAction()

            for line in range(line_from, line_to + 1):
                editor.unindent(line)

            editor.endUndoAction()

        else:

            editor.unindent(line)

    def check_for_existing_comments(self, line_from, line_to, comment_string_begin):
        """
        Checks if a comment exists already in the line
        :param line_from:
        :param line_to:
        :param comment_string_begin:
        :return: number of lines where comment exists already
        """

        comment_string_begin = comment_string_begin.strip()

        editor = self.getActiveEditor()

        num_existing_comments = 0

        for line_no in range(line_from, line_to+1):
            line_text = editor.text(line_no)

            orig_line_text_length = len(line_text)

            index_of = line_text.find(comment_string_begin)
            is_line_empty = not line_text.strip()
            if index_of >= 0 or is_line_empty:
                num_existing_comments += 1

        return num_existing_comments

    def get_comment_string_begin(self, editor) -> tuple:
        """
        Returns comment string begin and truncated comment begin
        :param editor:
        :return:
        """
        comment_string_begin = self.commentStyleDict[editor][0]

        comment_string_begin_trunc = comment_string_begin.strip()  # comments without white spaces

        if comment_string_begin_trunc == "REM":
            # comments which begin with a word  - e.g. REM should not be truncated
            comment_string_begin_trunc = comment_string_begin

        return comment_string_begin, comment_string_begin_trunc

    def get_comment_string_end(self, editor) -> tuple:
        """
        Returns comment string end and truncated comment end
        :param editor:
        :return:
        """
        comment_string_end = ''

        if self.commentStyleDict[editor][1]:
            comment_string_end = self.commentStyleDict[editor][1]

        comment_string_end_trunc = comment_string_end.strip()

        return comment_string_end, comment_string_end_trunc


    def block_uncomment(self):

        """

            slot called when block-uncommenting code

        """

        editor = self.getActiveEditor()

        if not self.commentStyleDict[editor][0]:  # this language does not allow comments

            return

        # comment_string_begin = self.commentStyleDict[editor][0]
        #
        # comment_string_begin_trunc = comment_string_begin.strip()  # comments without white spaces
        #
        # if comment_string_begin_trunc == "REM":
        #     # comments which begin with a word  - e.g. REM should not be truncated
        #     comment_string_begin_trunc = comment_string_begin

        comment_string_begin, comment_string_begin_trunc = self.get_comment_string_begin(editor=editor)
        comment_string_end, comment_string_end_trunc = self.get_comment_string_end(editor=editor)

        if editor.hasSelectedText():

            line_from, index_from, line_to, index_to = editor.getSelection()

            if index_to == 0:
                line_to -= 1

            first_line_begin_comment_length = 0

            last_line_begin_comment_length = 0

            editor.beginUndoAction()

            for line in range(line_from, line_to + 1):

                begin_comment_length, end_comment_length = self.uncomment_line(line, comment_string_begin,
                                                                               comment_string_begin_trunc,
                                                                               comment_string_end,
                                                                               comment_string_end_trunc)

                if line == line_from:
                    first_line_begin_comment_length = begin_comment_length

                if line == line_to:
                    last_line_begin_comment_length = begin_comment_length

            editor.endUndoAction()

            # restoring selection

            if index_to == 0:

                line_to += 1

                editor.setSelection(line_from, index_from - first_line_begin_comment_length, line_to, index_to)

            else:

                editor.setSelection(line_from, index_from - first_line_begin_comment_length, line_to,

                                    index_to - last_line_begin_comment_length)

        else:

            # Uncomment current line

            line, index = editor.getCursorPosition()

            editor.beginUndoAction()

            self.uncomment_line(line, comment_string_begin, comment_string_begin_trunc, comment_string_end,

                                comment_string_end_trunc)

            editor.endUndoAction()

    @staticmethod
    def is_xml_comment(comment: str):
        """
        Checks if comment element (begin or end is XML comment)
        :param comment:
        :return:
        """
        return comment.strip() in ['<!--', '-->']

    @staticmethod
    def is_xml_proper_begin(line):
        """
        Checks if xml line has proper begin character
        :param line:
        :return:
        """
        try:
            return line.lstrip()[0] == '<'
        except IndexError:
            return False

    @staticmethod
    def is_xml_proper_end(line):
        """
        Checks if xml line has proper begin character
        :param line:
        :return:
        """
        try:
            return line.rstrip()[-1] == '>'
        except IndexError:
            return False

    @staticmethod
    def insert_string(str_to_insert, target_str, pos):
        """
        Inserts string into target string
        :param str_to_insert:
        :param target_str:
        :param pos:
        :return:
        """
        try:
            final_str = target_str[:pos] + str_to_insert + target_str[pos:]
        except IndexError:
            final_str = target_str

        return final_str

    def uncomment_line(self, line, comment_string_begin, comment_string_begin_trunc, comment_string_end,
                       comment_string_end_trunc):
        """
        helper function called during block-uncommenting of the code to uncomment single line of code
        """

        editor = self.getActiveEditor()

        comments_found = False

        line_text = editor.text(line)

        orig_line_text_length = len(line_text)

        index_of = line_text.find(comment_string_begin)

        begin_comment_length = 0

        end_comment_length = 0

        multiple_opening_comments_found = self.check_if_multiple_begin_comments_already_exists(
            line_text=line_text,
            comment_string_begin=comment_string_begin)

        # processing beginning of the line
        if index_of != -1:

            line_text = remove_n_chars(line_text, index_of, len(comment_string_begin))
            if self.is_xml_comment(comment_string_begin_trunc):
                nested_comment_extras = ['&lt;!&ndash; ',  '&lt;!&ndash;']
                nested_comment_extra = nested_comment_extras[0]
                nested_comment_pos = -1

                for nested_comment_extra_local in nested_comment_extras:
                    nested_comment_pos = line_text.find(nested_comment_extra_local)
                    if nested_comment_pos != -1:
                        nested_comment_extra = nested_comment_extra_local
                        break

                if nested_comment_pos != -1:
                    line_text = remove_n_chars(line_text, nested_comment_pos, len(nested_comment_extra))
                    pos = self.first_non_whitespace_pos(line_text)
                    line_text = line_text[:pos] + comment_string_begin + line_text[pos:]
                else:
                    if not self.is_xml_proper_begin(line_text):
                        pos = self.first_non_whitespace_pos(line_text)
                        line_text = self.insert_string(str_to_insert='<', target_str=line_text, pos=pos)

            begin_comment_length = len(comment_string_begin)

            comments_found = True

        else:

            index_of = line_text.find(comment_string_begin_trunc)

            if index_of != -1:

                line_text = remove_n_chars(line_text, index_of, len(comment_string_begin_trunc))

                if self.is_xml_comment(comment_string_begin_trunc):
                    nested_comment_extras = ['&lt;!&ndash; ', '&lt;!&ndash;']
                    nested_comment_extra = nested_comment_extras[0]
                    nested_comment_pos = -1

                    for nested_comment_extra_local in nested_comment_extras:
                        nested_comment_pos = line_text.find(nested_comment_extra_local)
                        if nested_comment_pos != -1:
                            nested_comment_extra = nested_comment_extra_local
                            break

                    if nested_comment_pos != -1:
                        line_text = remove_n_chars(line_text, nested_comment_pos, len(nested_comment_extra))
                        pos = self.first_non_whitespace_pos(line_text)
                        line_text = line_text[:pos] + comment_string_begin + line_text[pos:]
                    else:
                        if not self.is_xml_proper_begin(line_text):
                            pos = self.first_non_whitespace_pos(line_text)
                            line_text = self.insert_string(str_to_insert='<', target_str=line_text, pos=pos)

                begin_comment_length = len(comment_string_begin_trunc)

                comments_found = True

        if len(comment_string_end):

            # processing end of the line
            last_index_of = line_text.rfind(comment_string_end)

            if not multiple_opening_comments_found:
                # we only remove closing comment when we have a single opening comment
                if last_index_of != -1:
                    line_text = remove_n_chars(line_text, last_index_of, len(comment_string_end))
                    if self.is_xml_comment(comment_string_end_trunc):

                        nested_comment_extras = [' &ndash;&gt;', '&ndash;&gt;']
                        nested_comment_extra = nested_comment_extras[0]
                        nested_comment_pos = -1

                        for nested_comment_extra_local in nested_comment_extras:
                            nested_comment_pos = line_text.rfind(nested_comment_extra_local)
                            if nested_comment_pos != -1:
                                nested_comment_extra = nested_comment_extra_local
                                break

                        if nested_comment_pos != -1:
                            line_text = remove_n_chars(line_text, nested_comment_pos, len(nested_comment_extra))
                            pos = self.first_end_whitespace(line_text)
                            line_text = line_text[:pos] + comment_string_end + line_text[pos:]
                        if not self.is_xml_proper_end(line_text):
                            pos = self.first_end_whitespace(line_text)
                            line_text = self.insert_string(str_to_insert='>', target_str=line_text, pos=pos)
                    end_comment_length = len(comment_string_end)

                    comments_found = True

                else:

                    last_index_of = line_text.rfind(comment_string_end_trunc)

                    if last_index_of != -1:

                        line_text = remove_n_chars(line_text, last_index_of, len(comment_string_end_trunc))
                        if self.is_xml_comment(comment_string_end_trunc):

                            nested_comment_extras = [' &ndash;&gt;', '&ndash;&gt;']
                            nested_comment_extra = nested_comment_extras[0]
                            nested_comment_pos = -1

                            for nested_comment_extra_local in nested_comment_extras:
                                nested_comment_pos = line_text.rfind(nested_comment_extra_local)
                                if nested_comment_pos != -1:
                                    nested_comment_extra = nested_comment_extra_local
                                    break

                            if nested_comment_pos != -1:
                                line_text = remove_n_chars(line_text, nested_comment_pos, len(nested_comment_extra))
                                pos = self.first_end_whitespace(line_text)
                                line_text = line_text[:pos] + comment_string_end + line_text[pos:]
                            else:
                                if not self.is_xml_proper_end(line_text):
                                    pos = self.first_end_whitespace(line_text)
                                    line_text = self.insert_string(str_to_insert='>', target_str=line_text,
                                                               pos=pos)
                        end_comment_length = len(comment_string_end_trunc)
                        comments_found = True

        if comments_found:

            eol_pos = len(line_text)

            if line_text[eol_pos - 2] == "\r" or line_text[eol_pos - 2] == "\n":

                # second option is just in case - checking if we are dealin with CR LF or simple CR or LF end of line
                editor.setSelection(line, 0, line, orig_line_text_length - 1)

            else:

                editor.setSelection(line, 0, line, orig_line_text_length)

            editor.removeSelectedText()

            editor.insertAt(line_text, line, 0)

        return begin_comment_length, end_comment_length

    def find(self):

        """

            shows find/replace dialog popup

        """

        if self.findDialogForm:
            # this should deal with OSX issues

            self.findDialogForm.show()

            self.findDialogForm.raise_()

            self.findDialogForm.setFocus()

            self.findDialogForm.activateWindow()

            #             self.findDialogForm.show()

            return

        self.findDialogForm = FindAndReplaceDlg("", self)

        self.findDialogForm.setFindAndReaplceHistory(self.findAndReplaceHistory)

        # putting highlighted text into findLineEdit

        editor = self.getActiveEditor()

        self.findDialogForm.searchingSignal.connect(self.findNext)

        self.findDialogForm.replacingSignal.connect(self.replaceNext)

        self.findDialogForm.replacingAllSignal.connect(self.replaceAll)

        self.findDialogForm.initializeDialog(self.findAndReplaceHistory)  # resizes widget among other things

        # if editor.hasSelectedText():

        # self.findDialogForm.findLineEdit.setText(editor.selectedText())

        # find in files

        self.findDialogForm.searchingSignalIF.connect(self.findInFiles)

        self.findDialogForm.replacingSignalIF.connect(self.replaceInFiles)

        # find All in All open Docs

        self.findDialogForm.searchingAllInAllOpenDocsSignal.connect(self.findInFiles)

        # replace All in All open Docs

        self.findDialogForm.replacingAllInOpenDocsSignal.connect(self.replaceInFiles)

        self.findDialogForm.show()

    def findInFiles(self, _text, _filters, _directory, _mode=ALL_IN_FILES):

        """

            searches for _text in all files of _filters type in the _directory. It performs search in

            currentDocument (_mode=ALL_IN_CURRENT_DOC), all open files (_mode=ALL_IN_ALL_OPEN_DOCS) or all files (default mode)

            specified by _filters,_directory

        """

        _text = str(_text)

        _filters = str(_filters)

        _directory = str(_directory)

        if _directory.rstrip() != '':

            if not exists(_directory) or not isdir(_directory):
                ret = QtWidgets.QMessageBox.warning(self, "Directory Error",

                                                    'Cannot search files in directory ' + _directory + ' because it does not exist')

                return

        self.findDialogForm.setButtonsEnabled(False)

        dbgMsg("findInFiles")

        dbgMsg("search parameters", _text, " ", _filters, " ", _directory)

        re_flag = False

        if str(self.findDialogForm.syntaxComboBoxIF.currentText()) == "Regular expression":
            re_flag = True

        newSearchFlag = self.findAndReplaceHistory.newSearchParametersIF(_text, _filters, _directory)

        # constructing the list of files to be searched based on directory and filters input

        filters = str(_filters).split()

        dbgMsg(filters)

        matches = []

        if _mode == ALL_IN_FILES:

            for filter in filters:

                if self.findDialogForm.inAllSubFoldersCheckBoxIF.isChecked():

                    for root, dirnames, filenames in os.walk(str(_directory)):

                        for filename in fnmatch.filter(filenames, filter):
                            matches.append(os.path.join(root, filename))

                else:

                    root = str(_directory)

                    for filename in os.listdir(root):

                        if fnmatch.fnmatch(filename, filter):
                            matches.append(os.path.join(root, filename))

        # dbgMsg("matches=",matches)

        foundFiles = []

        if _mode == ALL_IN_FILES:

            foundFiles = self.findFiles(matches, _text, re_flag)

        elif _mode == ALL_IN_ALL_OPEN_DOCS:

            foundFiles = self.findAllInOpenDocs(_text, re_flag)

        elif _mode == ALL_IN_CURRENT_DOC:

            foundFiles = self.findAllInOpenDocs(_text, re_flag, True)

        else:

            self.findDialogForm.setButtonsEnabled(True)

            return

        findInFilesFormatter = FindInFilesResults()  # use  empty FindInFilesResults object as a formatter

        self.findDisplayWidget.addNewFindInFilesResults(findInFilesFormatter.produceSummaryRepr(foundFiles, _text))

        if not self.showFindInFilesDockAct.isChecked():
            self.showFindInFilesDockAct.trigger()  # calling toggle does not emit triggered signal and thus action slot is not called. calling trigger does the trick

        self.findDialogForm.setButtonsEnabled(True)

    def findAllInOpenDocs(self, _text, _reFlag=False, _inCurrentDoc=False):

        """

            searches for text/regex in open documents or in current document (depending on _inCurrentDoc flag)

        """

        # progress dialog not necessary here

        foundFiles = []

        editorList = []

        if _inCurrentDoc:

            activePanel = self.getActivePanel()

            editorList.append(activePanel.currentWidget())

        else:

            editorList = self.getEditorList()

        # print "editorList=",editorList

        findText = _text  # a new copy of a textTo Find

        if _reFlag:
            # here I will replace ( with \( and vice versa - to be consistent with  regex convention

            # findText = re.escape(findText)

            findText = self.swapEscaping(findText, "(")

            findText = self.swapEscaping(findText, ")")

        for editor in editorList:

            currentLine, currentIndex = editor.getCursorPosition()  # record current cursor position

            filename = self.getEditorFileName(editor)

            if filename == '':
                # we will use tab label as a filename for unsaved documents

                filename = editor.panel.tabText(editor.panel.indexOf(editor))

                # continue # we do not allow searching in the unsaved files

            foundFlag = editor.findFirst(findText, _reFlag, \
 \
                                         self.findDialogForm.caseCheckBox.isChecked(), \
 \
                                         self.findDialogForm.wholeCheckBox.isChecked(), \
 \
                                         False, \
 \
                                         True, 0, 0, False)

            findResults = None

            if foundFlag:
                foundFiles.append(FindInFilesResults(filename, _text, True))

                findResults = foundFiles[-1]

            while foundFlag:
                line, index = editor.getCursorPosition()

                # dbgMsg("FOUND OCCURENCE IN LINE ",line)

                findResults.addLineWithText(line, editor.text(line))

                foundFlag = editor.findNext()

            editor.setCursorPosition(currentLine, currentIndex)  # restore cursor position

        # print "foundFiles=",foundFiles

        return foundFiles

    def findFiles(self, _files, _text, _reFlag=False):

        """

            performs actual search for _text(possibly regex depending on _reFlag) in _files

        """

        progressDialog = QtWidgets.QProgressDialog(self)

        progressDialog.setCancelButtonText("&Cancel")

        numberOfFiles = len(_files)

        progressDialog.setRange(0, numberOfFiles)

        progressDialog.setWindowTitle("Find Text in Files")

        # progressDialog.raise_()

        foundFiles = []

        i = 1

        # findText = QString(_text)  # a new copy of a textTo Find

        findText = _text  # a new copy of a textTo Find

        if _reFlag:
            # here I will replace ( with \( and vice versa - to be consistent with  regex convention

            # findText = re.escape(findText)

            findText = self.swapEscaping(findText, "(")

            findText = self.swapEscaping(findText, ")")

        for filename in _files:

            # dbgMsg("SEARCHING ", filename)

            progressDialog.setValue(i)

            progressDialog.setLabelText("Searching file number %d of %d..." % (i, numberOfFiles))

            QtWidgets.qApp.processEvents()

            if progressDialog.wasCanceled():
                break

            inFile = QtCore.QFile(filename)

            if inFile.open(QtCore.QIODevice.ReadOnly):

                stream = QtCore.QTextStream(inFile)

                textEditLocal = QsciScintillaCustom(self)

                textEditLocal.setText(stream.readAll())

                foundFlag = textEditLocal.findFirst(findText, _reFlag, \
 \
                                                    self.findDialogForm.caseCheckBoxIF.isChecked(), \
 \
                                                    self.findDialogForm.wholeCheckBoxIF.isChecked(), \
 \
                                                    False, \
 \
                                                    True, 0, 0, False)

                findResults = None

                if foundFlag:
                    foundFiles.append(FindInFilesResults(filename, _text))

                    findResults = foundFiles[-1]

                while foundFlag:
                    line, index = textEditLocal.getCursorPosition()

                    # dbgMsg("FOUND OCCURENCE IN LINE ",line)

                    findResults.addLineWithText(line, textEditLocal.text(line))

                    foundFlag = textEditLocal.findNext()

            i += 1

        progressDialog.close()

        return foundFiles

    def replaceInFiles(self, _text, _replaceText, _filters, _directory, _mode=ALL_IN_FILES):

        """

            replaces _text with _replaceText in files with extensions specified by  _filters  stored in _directory.

            _mode is used to initiate search either in all files (_mode=ALL_IN_FILES) or in open documents (_mode=ALL_IN_ALL_OPEN_DOCS)

        """

        # dbgMsg("search parameters",_text," ", _filters," ",_directory)

        message = "About to replace all occurences of <b>\"%s\"</b>  <br> in ALL <b>\"%s\"</b> files inside directory:<br> %s  <br> Proceed?" % (

            _text, _filters, _directory)

        ret = QtWidgets.QMessageBox.warning(self, "Replace in Files",

                                            message,

                                            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)

        if ret == QtWidgets.QMessageBox.Cancel:
            return

        reFlag = False

        if str(self.findDialogForm.syntaxComboBoxIF.currentText()) == "Regular expression":
            reFlag = True

        newSearchFlag = self.findAndReplaceHistory.newReplaceParametersIF(_text, _replaceText, _filters, _directory)

        # self.findDialogForm.initializeAllSearchLists(self.findAndReplaceHistory)

        self.findDialogForm.setButtonsEnabled(False)

        # constructing the list of files to be searched based on directory and filters input

        filters = str(_filters).split()

        dbgMsg(filters)

        matches = []

        if _mode == ALL_IN_FILES:

            for filter in filters:

                if self.findDialogForm.inAllSubFoldersCheckBoxIF.isChecked():

                    for root, dirnames, filenames in os.walk(str(_directory)):

                        for filename in fnmatch.filter(filenames, filter):
                            matches.append(os.path.join(root, filename))

                else:

                    root = str(_directory)

                    for filename in os.listdir(root):

                        if fnmatch.fnmatch(filename, filter):
                            matches.append(os.path.join(root, filename))

        # dbgMsg("matches=",matches)

        replaceInFilesData = []

        if _mode == ALL_IN_FILES:

            replaceInFilesData = self.processReplaceInFiles(matches, _text, _replaceText, reFlag)

        elif _mode == ALL_IN_ALL_OPEN_DOCS:

            replaceInFilesData = self.processReplaceInAllOpenDocs(_text, _replaceText, reFlag)

        else:

            self.findDialogForm.setButtonsEnabled(True)

            return

        try:

            message = "Replaced %s occurences of \"<b>%s</b>\" in %s files" % (

                str(replaceInFilesData[0]), _text, str(replaceInFilesData[1]))

            ret = QtWidgets.QMessageBox.information(self, "Replace in Files",

                                                    message,

                                                    QtWidgets.QMessageBox.Ok)



        except IndexError:

            pass

        self.findDialogForm.setButtonsEnabled(True)

    def processReplaceInAllOpenDocs(self, _text, _replaceText, _reFlag):

        """

            actually performs replacements of _text (uses regex criteria is _reFlag==True) with _replaceText in all open documents

        """

        # progress dialog not necessary here

        replaceInFilesData = []

        fileCounter = 0

        substitutionCounter = 0

        editorList = self.getEditorList()

        findText = _text  # a new copy of a textTo Find

        if _reFlag:
            # here I will replace ( with \( and vice versa - to be consistent with  regex convention

            # findText = re.escape(findText)

            findText = self.swapEscaping(findText, "(")

            findText = self.swapEscaping(findText, ")")

        for editor in editorList:

            currentLine, currentIndex = editor.getCursorPosition()  # record current cursor position

            filename = self.getEditorFileName(editor)

            if filename == '':
                continue  # we do not allow searching or replacing in the unsaved files

            foundFlag = editor.findFirst(findText, _reFlag, \
 \
                                         self.findDialogForm.caseCheckBox.isChecked(), \
 \
                                         self.findDialogForm.wholeCheckBox.isChecked(), \
 \
                                         False, \
 \
                                         True, 0, 0, False)

            newFileCountedFlag = False

            editor.beginUndoAction()

            while foundFlag:

                line, index = editor.getCursorPosition()

                editor.replace(_replaceText)

                if not newFileCountedFlag:
                    fileCounter += 1

                    newFileCountedFlag = True

                substitutionCounter += 1

                foundFlag = editor.findFirst(findText, _reFlag, \
 \
                                             self.findDialogForm.caseCheckBox.isChecked(), \
 \
                                             self.findDialogForm.wholeCheckBox.isChecked(), \
 \
                                             False)

            editor.endUndoAction()

            editor.setCursorPosition(currentLine, currentIndex)  # restore cursor position

        replaceInFilesData = [substitutionCounter, fileCounter]

        return replaceInFilesData

        # no progress dialog is necessary

    def processReplaceInFiles(self, _files, _text, _replaceText, _reFlag=False):

        """

            actually performs replacements of _text (uses regex criteria is _reFlag==True) with _replaceText in documents listed in _files parameter

        """

        # have to deal with files which are currently open - enable undo action and use open editor for them

        # dbgMsg(warning before execution replace in files)

        progressDialog = QtWidgets.QProgressDialog(self)

        progressDialog.setCancelButtonText("&Cancel")

        numberOfFiles = len(_files)

        progressDialog.setRange(0, numberOfFiles)

        progressDialog.setWindowTitle("Replacing Text in Files")

        findText = _text  # a new copy of a textTo Find

        if _reFlag:
            # here I will replace ( with \( and vice versa - to be consistent with  regex convention

            # findText = re.escape(findText)

            findText = self.swapEscaping(findText, "(")

            findText = self.swapEscaping(findText, ")")

        replaceInFilesData = []

        i = 1

        # this is used to construct dict of open files

        openFileDict = self.getFileNameToEditorWidgetMap()

        fileCounter = 0

        substitutionCounter = 0

        for filename in _files:

            # dbgMsg("SEARCHING ", filename)

            progressDialog.setValue(i)

            progressDialog.setLabelText("Searching file number %d of %d..." % (i, numberOfFiles))

            QtWidgets.qApp.processEvents()

            if progressDialog.wasCanceled():
                break

            textEditLocal = None

            usingOpenEditor = False

            openFileForReplaceFlag = False

            fileNameNormalized = os.path.abspath(str(filename))

            if fileNameNormalized in list(openFileDict.keys()):

                textEditLocal = openFileDict[fileNameNormalized]

                textEditLocal.setCursorPosition(0, 0)  # have to move cursor to the begining of the document

                usingOpenEditor = True

            else:

                inFile = QtCore.QFile(filename)

                if not inFile.open(

                        QtCore.QIODevice.ReadWrite):  # this will take care of write protected files - we will not process them

                    continue

                stream = QtCore.QTextStream(inFile)

                textEditLocal = QsciScintillaCustom(self)

                textEditLocal.setText(stream.readAll())

            foundFlag = textEditLocal.findFirst(findText, _reFlag, \
 \
                                                self.findDialogForm.caseCheckBoxIF.isChecked(), \
 \
                                                self.findDialogForm.wholeCheckBoxIF.isChecked(), \
 \
                                                False, \
 \
                                                True, 0, 0, False)

            findResults = None

            newFileCountedFlag = False

            if usingOpenEditor:
                textEditLocal.beginUndoAction()

            while foundFlag:

                line, index = textEditLocal.getCursorPosition()

                textEditLocal.replace(_replaceText)

                if not newFileCountedFlag:
                    fileCounter += 1

                    newFileCountedFlag = True

                substitutionCounter += 1

                foundFlag = textEditLocal.findFirst(findText, _reFlag, \
 \
                                                    self.findDialogForm.caseCheckBoxIF.isChecked(), \
 \
                                                    self.findDialogForm.wholeCheckBoxIF.isChecked(), \
 \
                                                    False)

            if usingOpenEditor:

                textEditLocal.endUndoAction()

                if newFileCountedFlag:
                    self.saveFile(filename, textEditLocal)  # using saveFile function for open editor

                    # heve to deactivate modification time sensing

                    textEditLocal.modificationChanged.disconnect(

                        self.textChangedHandlers[textEditLocal].handleModificationChanged)

                    self.setEditorFileModificationTime(textEditLocal, os.path.getmtime(str(filename)))

                    # heve to reactivate modification time sensing

                    textEditLocal.modificationChanged.connect(

                        self.textChangedHandlers[textEditLocal].handleModificationChanged)

            else:

                inFile.close()  # before writing we close the file - alternatively we may move file pointer to the begining

                if newFileCountedFlag and inFile.open(QtCore.QIODevice.WriteOnly):
                    outf = QtCore.QTextStream(inFile)

                    outf << textEditLocal.text()

            i += 1

        progressDialog.close()

        replaceInFilesData = [substitutionCounter, fileCounter]

        return replaceInFilesData

    def swapEscaping(self, _str, _char):

        """

            This fcn escapes character _str and if it is escaped it unescapes it

            It does not look like it is equivalent to re.excape(_str)

        """

        dbgMsg("string=", _str)

        idx = 0

        while idx >= 0:

            try:

                idx = _str[idx:].index(_char) + idx

            except ValueError:

                break

            # idx = _str.indexOf(_char, idx)

            # pd("Found index in position ",idx)

            if idx == 0:

                # _str.insert(idx, "\\")

                _str = _str[:idx] + '\\' + _str[idx + 1:]

                idx += 2

            elif idx >= 1:

                if _str[idx - 1] == '\\':

                    # if QString(_str.at(idx - 1)) == "\\":

                    _str = _str[:idx - 1] + _str[idx:]



                else:

                    _str = _str[:idx] + '\\' + _str[idx + 1:]

                    # _str.insert(idx, "\\")

                    idx += 2

        return _str

    def findNext(self, _text):

        """

            slot called whe user selects FindNext or presses F3

        """

        editor = self.getActiveEditor()

        self.findDialogForm.setButtonsEnabled(False)

        reFlag = False

        if str(self.findDialogForm.syntaxComboBox.currentText()) == "Regular expression":
            reFlag = True

        self.textToFind = _text

        newSearchFlag = self.findAndReplaceHistory.newSearchParameters(_text, reFlag,

                                                                       self.findDialogForm.caseCheckBox.isChecked(),

                                                                       self.findDialogForm.wholeCheckBox.isChecked(),

                                                                       True,

                                                                       self.findDialogForm.inSelectionBox.isChecked())

        findText = self.findAndReplaceHistory.textToFind  # a new copy of a textTo Find

        if self.findAndReplaceHistory.re:
            # here I will replace ( with \( and vice versa - to be consistent with  regex convention

            # findText = re.escape(findText)

            findText = self.swapEscaping(findText, "(")

            findText = self.swapEscaping(findText, ")")

            pass

        if newSearchFlag:

            foundFlag = editor.findFirst(findText, self.findAndReplaceHistory.re, self.findAndReplaceHistory.cs,

                                         self.findAndReplaceHistory.wo, self.findAndReplaceHistory.wrap)

            if not foundFlag:
                message = "Cannot find \"<b>%s</b>\"" % self.findAndReplaceHistory.textToFind

                ret = QtWidgets.QMessageBox.information(self, "Find",

                                                        message,

                                                        QtWidgets.QMessageBox.Ok)

        else:

            # for some reason findNext does not work after undo action...

            foundFlag = editor.findFirst(findText, self.findAndReplaceHistory.re, self.findAndReplaceHistory.cs,

                                         self.findAndReplaceHistory.wo, self.findAndReplaceHistory.wrap)

            if not foundFlag:
                message = "Cannot find \"<b>%s</b>\"" % self.findAndReplaceHistory.textToFind

                ret = QtWidgets.QMessageBox.information(self, "Find",

                                                        message,

                                                        QtWidgets.QMessageBox.Ok)

        self.findDialogForm.setButtonsEnabled(True)

    def findNextSimple(self):

        """

            this slot does not work as advertised in QScintilla documentation - not used

        """

        editor = self.getActiveEditor()

        findText = str(self.findAndReplaceHistory.textToFind)  # a new copy of a textTo Find

        if self.findAndReplaceHistory.re:
            # here I will replace ( with \( and vice versa - to be consistent with  regex convention

            # findText = re.escape(findText)

            findText = self.swapEscaping(findText, "(")

            findText = self.swapEscaping(findText, ")")

        foundFlag = editor.findFirst(findText, self.findAndReplaceHistory.re, self.findAndReplaceHistory.cs,

                                     self.findAndReplaceHistory.wo, self.findAndReplaceHistory.wrap)

        return

        # old code

        # editor=self.getActiveEditor()

        # editor.findNext()

    def replaceNext(self, _text, _replaceText):

        """

            slot called whe user selects Replace repetidly in Find/replace dialog popup

        """

        self.findDialogForm.setButtonsEnabled(False)

        editor = self.getActiveEditor()

        re_flag = False

        in_selection_flag = self.findDialogForm.inSelectionBox.isChecked()

        if str(self.findDialogForm.syntaxComboBox.currentText()) == "Regular expression":
            re_flag = True

        new_replace_flag = self.findAndReplaceHistory.newReplaceParameters(_text, _replaceText, re_flag,
                                                                           self.findDialogForm.caseCheckBox.isChecked(),
                                                                           self.findDialogForm.wholeCheckBox.isChecked(),
                                                                           True, in_selection_flag)

        find_text = str(_text)

        if self.findAndReplaceHistory.re:
            # here I will replace ( with \( and vice versa - to be consistent with  regex convention

            # findText = re.escape(findText)

            find_text = self.swapEscaping(find_text, "(")

            find_text = self.swapEscaping(find_text, ")")

        line_from, index_from, line_to, index_to = editor.getSelection()

        if new_replace_flag:

            found_flag = editor.findFirst(find_text, self.findAndReplaceHistory.re, self.findAndReplaceHistory.cs,
                                          self.findAndReplaceHistory.wo, False, True, line_from, index_from)

            if not found_flag:
                message = "Cannot find \"<b>%s</b>\"" % self.findAndReplaceHistory.textToFind

                ret = QtWidgets.QMessageBox.information(self, "Replace", message, QtWidgets.QMessageBox.Ok)

                self.findDialogForm.setButtonsEnabled(True)

                return

            editor.replace(self.findAndReplaceHistory.replaceText)

        else:

            # for some reason there are problem using findNext together with replace - since editor is expanding consecutive calls

            # to replace (especially when replacing short string with long string) result in replace abbandoning changes because it may think there is not eanough space. findFirst works fine though

            found_flag = editor.findFirst(find_text, self.findAndReplaceHistory.re, self.findAndReplaceHistory.cs,

                                          self.findAndReplaceHistory.wo, True, True)

            # editor.SendScintilla(QsciScintilla.SCI_REPLACESEL,0, self.findAndReplaceHistory.replaceText)

            if not found_flag:
                message = "Cannot find \"<b>%s</b>\"" % self.findAndReplaceHistory.textToFind

                ret = QtWidgets.QMessageBox.information(self, "Replace", message, QtWidgets.QMessageBox.Ok)
                self.findDialogForm.setButtonsEnabled(True)

                return

            editor.replace(self.findAndReplaceHistory.replaceText)

        self.findDialogForm.setButtonsEnabled(True)

    def replaceAll(self, _text, _replaceText, _inSelectionFlag):

        """

            slot called when user clicks Replace All on Find/replace popup. _inSelection flag determines if replacement takes place in the entire document or only in the selected text

        """

        self.findDialogForm.setButtonsEnabled(False)

        editor = self.getActiveEditor()

        re_flag = False

        inSelectionFlag = _inSelectionFlag

        if str(self.findDialogForm.syntaxComboBox.currentText()) == "Regular expression":
            re_flag = True

        newReplaceFlag = self.findAndReplaceHistory.newReplaceParameters(_text, _replaceText, re_flag,
                                                                         self.findDialogForm.caseCheckBox.isChecked(),

                                                                         self.findDialogForm.wholeCheckBox.isChecked(),

                                                                         True, inSelectionFlag)

        substitutionCounter = 0

        findText = self.findAndReplaceHistory.textToFind  # a new copy of a textTo Find

        if self.findAndReplaceHistory.re:
            # here I will replace ( with \( and vice versa - to be consistent with  regex convention

            # findText = re.escape(findText)

            findText = self.swapEscaping(findText, "(")

            findText = self.swapEscaping(findText, ")")

        if inSelectionFlag and editor.hasSelectedText():

            # if editor.hasSelectedText():

            line_before, index_before = editor.getCursorPosition()

            line_from, index_from, line_to, index_to = editor.getSelection()

            index_to_original = index_to

            # print 'index_to_original=',index_to_original

            foundFlag = editor.findFirst(findText, \
 \
                                         self.findAndReplaceHistory.re, \
 \
                                         self.findAndReplaceHistory.cs, \
 \
                                         self.findAndReplaceHistory.wo, \
 \
                                         False, \
 \
                                         True, line_from, index_from, False)

            if not foundFlag:
                message = "Cannot find \"<b>%s</b>\"" % findText

                ret = QtWidgets.QMessageBox.information(self, "Replace All",

                                                        message,

                                                        QtWidgets.QMessageBox.Ok)

                # message="Cannot find \"<b>%s</b>\""%self.findAndReplaceHistory.textToFind

                # ret = QtWidgets.QMessageBox.information(self, "Replace All",

                # message,

                # QtWidgets.QMessageBox.Ok )

                self.findDialogForm.setButtonsEnabled(True)

                return

            line, index = editor.getCursorPosition()

            # print 'AFTER FIRST SEARCH line_to, index_to=',line_to, index_to

            editor.beginUndoAction()  # undo

            while foundFlag and (line < line_to or (line == line_to and index <= index_to)):

                lineBeforeReplace, indexBeforeReplace = editor.getCursorPosition()

                # print 'NEW FIND------------------------------'

                # print 'before  replace=',lineBeforeReplace,indexBeforeReplace

                editor.replace(self.findAndReplaceHistory.replaceText)

                lineAfterReplace, indexAfterReplace = editor.getCursorPosition()

                # print 'After replace=',lineAfterReplace,indexAfterReplace

                deltaLine = lineAfterReplace - lineBeforeReplace

                deltaIndex = indexAfterReplace - indexBeforeReplace

                substitutionCounter += 1

                foundFlag = editor.findFirst(findText, \
 \
                                             self.findAndReplaceHistory.re, \
 \
                                             self.findAndReplaceHistory.cs, \
 \
                                             self.findAndReplaceHistory.wo, \
 \
                                             False)

                line, index = editor.getCursorPosition()

                line_to += deltaLine

                # after finding new search phrase we decide whether to increase index_to (when found new phrase is in the same line as lineAfter) or whether to reset it to index_to_original if phrase is in the next line

                if line != lineAfterReplace:

                    index_to = index_to_original

                else:

                    index_to += deltaIndex

                    # print 'deltaLine,deltaindex=',deltaLine,deltaIndex

                    # print 'line_to,index_to=',line_to,index_to

                    # print '------------------------------'

            editor.endUndoAction()  # undo

            editor.setCursorPosition(line_before, index_before)

            index_mark = index_to_original

            if index_to < index_to_original:
                index_mark = index_to

            # print 'index_mark=',index_mark

            editor.setSelection(line_from, index_from, line_to, index_mark)

        elif not inSelectionFlag:

            line_from = 0

            index_from = 0

            line_before, index_before = editor.getCursorPosition()

            foundFlag = editor.findFirst(findText, \
 \
                                         self.findAndReplaceHistory.re, \
 \
                                         self.findAndReplaceHistory.cs, \
 \
                                         self.findAndReplaceHistory.wo, \
 \
                                         False, \
 \
                                         True, line_from, index_from, False)

            if not foundFlag:
                message = "Cannot find \"<b>%s</b>\"" % self.findAndReplaceHistory.textToFind

                ret = QtWidgets.QMessageBox.information(self, "Replace All",

                                                        message,

                                                        QtWidgets.QMessageBox.Ok)

                self.findDialogForm.setButtonsEnabled(True)

                return

                # previousLine,previousPos=editor.getCursorPosition()

            editor.beginUndoAction()  # undo

            while foundFlag:
                editor.replace(self.findAndReplaceHistory.replaceText)

                substitutionCounter += 1

                foundFlag = editor.findFirst(findText, \
 \
                                             self.findAndReplaceHistory.re, \
 \
                                             self.findAndReplaceHistory.cs, \
 \
                                             self.findAndReplaceHistory.wo, \
 \
                                             False)

            editor.endUndoAction()  # undo

            editor.setCursorPosition(line_before, index_before)

        message = "Replaced %s occurences of \"<b>%s</b>\"" % (str(substitutionCounter), _text)

        ret = QtWidgets.QMessageBox.information(self, "Replace in Files",

                                                message,

                                                QtWidgets.QMessageBox.Ok)

        self.findDialogForm.setButtonsEnabled(True)

    def marginClickedHandler(self, _margin, _line, _keyboardState):

        """

            slot  which toggles bookmark when margin gets clicked

        """

        editor = self.getActiveEditor()

        if _margin == 1:

            if editor.markersAtLine(_line) != self.bookmarkMask:  # check if there is marker in this liine

                marker = editor.markerAdd(_line, self.lineBookmark)

            else:  # otherwise remove bookmark

                editor.markerDelete(_line)

    def toggleBookmark(self):

        """

            slot which toggles bookmark when user puts bookmark using menu or F2

        """

        editor = self.getActiveEditor()

        line, index = editor.getCursorPosition()

        if editor.markersAtLine(line) != self.bookmarkMask:  # check if there is marker in this liine

            # if not add bookmark

            marker = editor.markerAdd(line, self.lineBookmark)

        else:  # otherwise remove bookmark

            editor.markerDelete(line)

    def goToNextBookmark(self):

        """

            slot - moves cursor to the next bookmark

        """

        editor = self.getActiveEditor()

        line, index = editor.getCursorPosition()

        lineNext = editor.markerFindNext(line, self.bookmarkMask)

        if lineNext == line:
            lineNext = editor.markerFindNext(line + 1, self.bookmarkMask)

        if lineNext == -1:

            lineNext = editor.markerFindNext(0, self.bookmarkMask)

            if lineNext == -1:
                return

        if lineNext != line and lineNext != -1:
            editor.setCursorPosition(lineNext, 0)

            return

    def goToPreviousBookmark(self):

        """

            slot - moves cursor to the previous bookmark

        """

        editor = self.getActiveEditor()

        line, index = editor.getCursorPosition()

        linePrevious = editor.markerFindPrevious(line, self.bookmarkMask)

        if linePrevious == line:
            linePrevious = editor.markerFindPrevious(line - 1, self.bookmarkMask)

        if linePrevious == -1:

            linePrevious = editor.markerFindPrevious(editor.lines(), self.bookmarkMask)

            if linePrevious == -1:
                return

        if linePrevious != line and linePrevious != -1:
            editor.setCursorPosition(linePrevious, 0)

            return

    def deleteAllBookmarks(self):

        """

            slot - removes all bookmarks from current doc

        """

        editor = self.getActiveEditor()

        editor.markerDeleteAll(self.lineBookmark)

    def goToLineShow(self, _line):

        """

            slot - displays go to line dialog

        """

        editor = self.getActiveEditor()

        self.goToLineDlg = GoToLineDlg(editor, self)

        self.goToLineDlg.show()

    def goToLine(self, _line):

        """

            moves cursor to specified _line

        """

        dbgMsg("GO TO LINE SLOT = ", _line)

        editor = self.getActiveEditor()

        editor.setCursorPosition(_line - 1, 0)

    def goToMatchingBrace(self):

        """

            moves cursor to matching brace

        """

        editor = self.getActiveEditor()

        editor.moveToMatchingBrace()

    def selectToMatchingBrace(self):

        """

            selects text to matching brace

        """

        editor = self.getActiveEditor()

        editor.selectToMatchingBrace()

    def configurationUpdate(self):

        """

            fcn handling in the configuration dialog

        """

        editor = self.getActiveEditor()

        configuration_dlg = ConfigurationDlg(editor, self)

        old_theme_name = self.currentThemeName

        if configuration_dlg.exec_():

            for key in list(self.configuration.updatedConfigs.keys()):
                dbgMsg("NEW SETTING = ", key, ":", self.configuration.updatedConfigs[key])

                configure_fcn = getattr(self, "configure" + key)

                configure_fcn(self.configuration.updatedConfigs[key])

        else:

            self.applyTheme(old_theme_name)

        self.checkActions()

    def configureRestoreTabsOnStartup(self, _flag):

        """

            fcn handling RestoreTabsOnStartup configuration change

        """

        self.configuration.setSetting("RestoreTabsOnStartup", _flag)

    def configureTheme(self, _themeName):

        """

            fcn handling theme configuration change

        """

        # print 'APPLYING _themeName=',_themeName

        self.currentThemeName = str(_themeName)

        self.configuration.setSetting("Theme", self.currentThemeName)

    def applyTheme(self, _themeName):

        self.currentThemeName = str(_themeName)

        for panel in self.panels:

            for i in range(panel.count()):
                editor = panel.widget(i)

                self.themeManager.applyThemeToEditor(self.currentThemeName, editor)

        # applying theme to FindInFiles widget

        self.themeManager.applyThemeToEditor(self.currentThemeName, self.findDisplayWidget)

        # applying 'Global override' style to all plugins

        # applying 'Default Style' style to all plugins

        # TODO enable it

        # self.pm.runForAllPlugins(function_name='applyStyleFromTheme',

        #                          argument_dict={'styleName': 'Default Style', 'themeName': self.currentThemeName})

    def configureBaseFontName(self, _name):

        """

            fcn handling BaseFontName configuration change

        """

        self.configuration.setSetting("BaseFontName", _name)

        self.baseFont = QFont(self.configuration.setting("BaseFontName"),

                              int(self.configuration.setting("BaseFontSize")))

        for panel in self.panels:

            for i in range(panel.count()):

                lexer = panel.widget(i).lexer()

                if lexer:
                    lexer.setFont(self.baseFont)

                panel.widget(i).setFont(self.baseFont)

    def configureBaseFontSize(self, _size):

        """

            fcn handling BaseFontSize configuration change

        """

        self.configuration.setSetting("BaseFontSize", int(_size))

        self.baseFont = QFont(self.configuration.setting("BaseFontName"),

                              int(self.configuration.setting("BaseFontSize")))

        for panel in self.panels:

            for i in range(panel.count()):

                lexer = panel.widget(i).lexer()

                if lexer:
                    lexer.setFont(self.baseFont)

                panel.widget(i).setFont(self.baseFont)

    def configureUseTabSpaces(self, _flag):

        """

            fcn handling UseTabSpaces configuration change

        """

        for panel in self.panels:

            for i in range(panel.count()):

                panel.widget(i).setIndentationsUseTabs(not _flag)

                if _flag:

                    panel.widget(i).setIndentationWidth(self.configuration.setting("TabSpaces"))

                else:

                    panel.widget(i).setIndentationWidth(

                        0)  # If width is 0 then the value returned by tabWidth() is used

    def configureTabSpaces(self, _value):

        """

            fcn handling TabSpaces configuration change

        """

        flag = self.configuration.setting("UseTabSpaces")

        for panel in self.panels:

            for i in range(panel.count()):

                panel.widget(i).setIndentationsUseTabs(not flag)

                if flag:

                    panel.widget(i).setIndentationWidth(_value)

                else:

                    panel.widget(i).setIndentationWidth(

                        0)  # If width is 0 then the value returned by tabWidth() is used

    def configureFoldText(self, _flag):

        """

            fcn handling FoldText configuration change

        """

        for panel in self.panels:

            for i in range(panel.count()):

                if _flag:

                    panel.widget(i).setFolding(QsciScintilla.BoxedTreeFoldStyle)  # 5 corresponds to BoxedTreeFoldStyle

                else:

                    panel.widget(i).setFolding(QsciScintilla.NoFoldStyle)  # no folding

    def configureDisplayWhitespace(self, _flag):

        """

            fcn handling DisplayWhitespace configuration change

        """

        for panel in self.panels:

            for i in range(panel.count()):

                if _flag:

                    panel.widget(i).setWhitespaceVisibility(QsciScintilla.SCWS_VISIBLEALWAYS)  # whitespace

                else:

                    panel.widget(i).setWhitespaceVisibility(QsciScintilla.SCWS_INVISIBLE)  # no whitespaces

    def configureDisplayEOL(self, _flag):

        """

            fcn handling DisplayEOL configuration change

        """

        for panel in self.panels:

            for i in range(panel.count()):
                panel.widget(i).setEolVisibility(_flag)

    def configureWrapLines(self, _flag):

        """

            fcn handling WrapLines configuration change

            on Mac setting word wrap can be very slow so it wrap lines is disabled for OSX

        """

        # on Mac setting word wrap can be very slow so it is best to avoid processing many documents at once

        if sys.platform.startswith('darwin'):
            return

        for panel in self.panels:

            for i in range(panel.count()):

                if _flag:

                    panel.widget(i).setWrapMode(QsciScintilla.WrapWord)

                else:

                    panel.widget(i).setWrapMode(QsciScintilla.WrapNone)

    def configureShowWrapSymbol(self, _flag):

        """

            fcn handling ShowWrapSymbol configuration change

        """

        for panel in self.panels:

            for i in range(panel.count()):

                if _flag:

                    panel.widget(i).setWrapVisualFlags(QsciScintilla.WrapFlagByText)

                else:

                    panel.widget(i).setWrapVisualFlags(QsciScintilla.WrapFlagNone)

    def configureTabGuidelines(self, _flag):

        """

            fcn handling TabGuidelines configuration change

        """

        for panel in self.panels:

            for i in range(panel.count()):
                panel.widget(i).setIndentationGuides(_flag)

    def configureDisplayLineNumbers(self, _flag):

        """

            fcn handling DisplayLineNumbers configuration change

        """

        for panel in self.panels:

            for i in range(panel.count()):
                editor = panel.widget(i)

                self.adjustLineNumbers(editor, _flag)
                self.checkActions()

    def fix_line_number_margin_width(self, editor):

        number_of_lines = editor.lines()

        number_of_digits = int(log(number_of_lines, 10)) + 2 if number_of_lines > 0 else 2
        editor.setMarginWidth(0, '0' * number_of_digits)


    def adjustLineNumbers(self, _editor, _flag):

        _editor.setMarginLineNumbers(0, _flag)
        if _flag:
            self.fix_line_number_margin_width(editor=_editor)
        else:
            _editor.setMarginWidth(0, '0' * 1)

    def configureEnableAutocompletion(self, _flag):

        """

            fcn handling EnableAutocompletion configuration change

        """

        for panel in self.panels:

            for i in range(panel.count()):

                if _flag:

                    panel.widget(i).setAutoCompletionThreshold(self.configuration.setting("AutocompletionThreshold"))

                    panel.widget(i).setAutoCompletionSource(QsciScintilla.AcsAll)

                else:

                    panel.widget(i).setAutoCompletionSource(QsciScintilla.AcsNone)

    def configureEnableQuickTextDecoding(self, _flag):

        """

            fcn handling EnableQuickTextDecoding configuration change

        """

        pass

    def configureAutocompletionThreshold(self, _value):

        """

            fcn handling AutocompletionThreshold configuration change

        """

        flag = self.configuration.setting("EnableAutocompletion")

        for panel in self.panels:

            for i in range(panel.count()):

                if flag:

                    panel.widget(i).setAutoCompletionThreshold(_value)

                    panel.widget(i).setAutoCompletionSource(QsciScintilla.AcsAll)

                else:

                    panel.widget(i).setAutoCompletionThreshold(_value)

                    panel.widget(i).setAutoCompletionSource(QsciScintilla.AcsNone)

    def newFile(self, _tabWidget=None):

        """

            slot which creqates new empty file and opens it in the active panel

        """

        # print "_tabWidget=",_tabWidget

        # if _tabWidget is not None:

        # print "HAVE NON ZERO TAB WIDGET=",_tabWidget

        activePanel = None

        # if  isinstance(_tabWidget,CustomTabWidget):

        if _tabWidget is not None and _tabWidget:

            activePanel = _tabWidget

            # print "ACTIVE TAB=",activePanel

        else:

            activePanel = self.getActivePanel()

        # print "activePanel=",activePanel

        # print "_tabWidget=",_tabWidget

        # am.setActionKeyboardShortcut("New","Ctrl+P")

        print("activePanel=", activePanel)

        textEditLocal = QsciScintillaCustom(self, activePanel)

        # print "activePanel=",activePanel

        activePanel.addTab(textEditLocal, QtGui.QIcon(':/icons/document-clean.png'),

                           "New Document " + str(activePanel.count() + 1))

        activePanel.setCurrentWidget(textEditLocal)

        self.setCurrentFile('')

        self.setEditorProperties(textEditLocal)

        self.commentStyleDict[activePanel.currentWidget()] = ['', '']

        self.setPropertiesInEditorList(activePanel.currentWidget(), '', 0, 'utf-8')

        # adding text Changed Handler

        self.textChangedHandlers[textEditLocal] = ChangedTextHandler(textEditLocal, self)

        editorIndex = activePanel.indexOf(textEditLocal)

        activePanel.widget(editorIndex).modificationChanged.connect(

            self.textChangedHandlers[textEditLocal].handleModificationChanged)

        activePanel.widget(editorIndex).textChanged.connect(self.textChangedHandlers[textEditLocal].handleChangedText)

        activePanel.widget(editorIndex).cursorPositionChanged.connect(self.handleCursorPositionChanged)

        # applygin theme to new document

        self.themeManager.applyThemeToEditor(self.currentThemeName, activePanel.widget(editorIndex))

    def __openRecentDirectory(self):

        '''

            slot handling sel;ection from File->RecentDirectories...

        '''

        action = self.sender()

        dirName = ''

        if isinstance(action, QAction):
            dirName = str(action.data().toString())

        fileNames = QtGui.QFileDialog.getOpenFileNames(self, "Open new file...", dirName, self.fileDialogFilters)

        self.add_item_to_configuration_string_list(self.configuration, "RecentDirectories", dirName)

        if fileNames.count():
            # extract path name and add it to settings

            sampleFileName = fileNames[0]

            dirName = os.path.abspath(os.path.dirname(str(sampleFileName)))

            self.add_item_to_configuration_string_list(self.configuration, "RecentDirectories", dirName)

            self.loadFiles(fileNames)

    def open(self):

        """

            slot handling file open dialog

            REMARK: on some linux distros you need to run manually ibus-setup to enable proper behavior of QFile dialog.

            Maybe there is a way to ensure it without this step - have to check it though

        """

        # get path to file in the current widget

        # REMARK: on some linux distros you need to run manually ibus-setup to enable proper behavior of QFile dialog.

        # Maybe there is a way to ensure it without this step - have to check it though

        dbgMsg("INSIDE OPEN")

        current_file_path = None

        try:
            editor = self.getActiveEditor()
            current_file_path = self.getEditorFileName(editor)

        except KeyError:

            pass

        if current_file_path:

            current_file_path = os.path.dirname(str(current_file_path))

            self.lastFileOpenPath = current_file_path

        else:

            current_file_path = self.lastFileOpenPath

        dbgMsg("THIS IS CURRENT PATH=", current_file_path)

        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open new file...", current_file_path,

                                                               self.fileDialogFilters)

        if len(file_names):
            # extract path name and add it to settings

            sampleFileName = file_names[0]

            dirName = os.path.abspath(os.path.dirname(str(sampleFileName)))

            self.add_item_to_configuration_string_list(self.configuration, "RecentDirectories", dirName)

            self.loadFiles(file_names)

    def closeTabIndex(self, _index):

        """

            closes tab with with index=_index

        """

        self.closeTab(_index)

    def closeTab(self, index=None, _askToSave=True, _panel=None):

        """

            closes tab with specified index, belonging to panel _panel (in _panel is None then active panel is used)

            Depending on _askToSave flag checks if document should be saved or skips this check

        """

        activePanel = None

        activePanel = self.getActivePanel()

        # print "1 activePanel=",activePanel

        sender = self.sender()

        if sender in self.panels:
            activePanel = sender

            # print "SENDER IS A TAB"

        if _panel:
            activePanel = _panel

            # if _panelIdx in (0,1):

            # activePanel=self.panels[_panelIdx]

            # else:

            # activePanel=self.getActivePanel()

            # activePanel=self.sender()

            # print "2 activePanel=",activePanel,

            # if activePanel==self.panels[0]:

            # print "activePanel is 0"

            # else:

            # print "activePanel is 1"

        # print "CLOSE TAB "

        dbgMsg("closing tab ", index)

        textEditLocal = None

        if index is not None and not isinstance(index,

                                                bool):  # ctrl+w calls closeTab with False bool type argument and we need to make sure that it is not converted to int index

            textEditLocal = activePanel.widget(index)

        else:

            textEditLocal = activePanel.currentWidget()

        if _askToSave:

            # print "textEditLocal=",textEditLocal

            reallyClose = self.maybeSave(textEditLocal)

            if not reallyClose:
                return

        if activePanel.count() == 1:

            if activePanel == self.panels[0] and not self.panels[1].count():

                # print "activePanel.count()==1 \n\n\n"

                # and not self.panels[1].count()

                # print "activePanel==self.panels[0] and not self.panels[1].count()\n\n"

                # self.insertEmptyDocument(activePanel)

                activePanel.currentWidget().clear()

                activePanel.currentWidget().setModified(

                    False)  # clearing document modifies its content but since no new text has been typed we set modified to false

                activePanel.setTabText(0, 'Empty Document')

                self.setCurrentFile('')

                self.commentStyleDict[activePanel.currentWidget()] = ['', '']

                if activePanel.currentWidget() in self.getEditorList():
                    # del self.fileDict[activePanel.currentWidget()]

                    self.removeEditor(activePanel.currentWidget())

                activePanel.setTabIcon(0, QtGui.QIcon(':/icons/document-clean.png'))

                # print "activePanel.count()=",activePanel.count()

            else:

                indexOfRemovedTab = activePanel.indexOf(textEditLocal)

                activePanel.removeTab(indexOfRemovedTab)

                try:

                    del self.commentStyleDict[textEditLocal]

                except LookupError as e:

                    pass

                newActiveIndex = 0

                if indexOfRemovedTab >= activePanel.count():
                    newActiveIndex = activePanel.count() - 1

                if activePanel.count():

                    activePanel.setCurrentWidget(activePanel.widget(newActiveIndex))

                    activePanel.widget(newActiveIndex).setFocus(Qt.MouseFocusReason)

                elif not self.panels[0].count() and not self.panels[1].count():

                    # print "self.panels[0]=",self.panels[0]

                    self.newFile(self.panels[0])

                    self.panels[0].show()

                    self.panels[1].hide()

                    self.activePanelWidget = self.panels[0]

                else:

                    activePanel.hide()

                    # once tab has no widgets the other one which still has widgets become current tab

                    if activePanel == self.panels[0]:

                        self.activePanelWidget = self.panels[1]

                    else:

                        self.activePanelWidget = self.panels[0]

                        # try:

                        # except KeyError,e: # not all documents will have lexer

                        # pass

                self.removeEditor(textEditLocal)

                # del self.fileDict[textEditLocal]

                if activePanel.currentWidget() in self.textChangedHandlers:

                    try:

                        del self.textChangedHandlers[textEditLocal]

                    except LookupError as e:

                        pass



        else:

            indexOfRemovedTab = activePanel.indexOf(textEditLocal)

            activePanel.removeTab(indexOfRemovedTab)

            try:

                del self.commentStyleDict[textEditLocal]

            except LookupError as e:

                pass

            newActiveIndex = 0

            if indexOfRemovedTab >= activePanel.count():
                newActiveIndex = activePanel.count() - 1

            if activePanel.count():

                activePanel.setCurrentWidget(activePanel.widget(newActiveIndex))

                activePanel.widget(newActiveIndex).setFocus(Qt.MouseFocusReason)

                # elif not self.tabWidget[0].count() and not self.tabWidget[1].count():

                # self.newFile(self.tabWidget[0])

                # self.tabWidget[0].show()

            else:

                activePanel.hide()

                # try:

                # except KeyError,e: # not all documents will have lexer

                # pass

            self.removeEditor(textEditLocal)

            # del self.fileDict[textEditLocal]

            if activePanel.currentWidget() in self.textChangedHandlers:

                try:

                    del self.textChangedHandlers[textEditLocal]

                except LookupError as e:

                    pass

    def foldAll(self):

        """

            slot - folds entire code

        """

        editor = self.getActiveEditor()

        editor.foldAll(True)

    def wrapLines(self, _flag):

        """

            slot - wraps/unwraps (_depending on _flag) lines in the active editor  - updates View Menu

        """

        editor = self.getActiveEditor()

        if not editor:
            return

        if not _flag:

            editor.setWrapMode(QsciScintilla.WrapNone)

        else:

            if sys.platform.startswith('darwin'):

                if not self.configuration.setting("DontShowWrapLinesWarning"):

                    msgBox = QMessageBox(QMessageBox.Warning, "Wrap Lines Warning",

                                         "<b>Wrap Line operation on OS X may take a long time<\b>")

                    dontAskCheckBox = QCheckBox("Do not display this warning again", msgBox)

                    dontAskCheckBox.blockSignals(True)

                    msgBox.addButton(dontAskCheckBox, QMessageBox.ActionRole)

                    # msgBox.setTitle("Wrap Lines Warning")

                    # msgBox.setText("<b>Wrap Line operation on OS X may take a long time<\b>");

                    msgBox.setInformativeText("Proceed?")

                    msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

                    msgBox.setDefaultButton(QMessageBox.No)

                    ret = msgBox.exec_()

                    if ret == QMessageBox.Yes:

                        self.configuration.setSetting("DontShowWrapLinesWarning", dontAskCheckBox.isChecked())

                    else:

                        self.configuration.setSetting("DontShowWrapLinesWarning", dontAskCheckBox.isChecked())

                        return

            editor.setWrapMode(QsciScintilla.WrapWord)

    def showWhitespaces(self, _flag):

        """

            slot - shows white spaces or hides them (_depending on _flag) in the active editor  - updates View Menu

        """

        editor = self.getActiveEditor()

        if not editor:
            return

        if _flag:

            editor.setWhitespaceVisibility(QsciScintilla.SCWS_VISIBLEALWAYS)

        else:

            editor.setWhitespaceVisibility(QsciScintilla.SCWS_INVISIBLE)

    def showEOL(self, _flag):

        """

            slot - shows or hides EOL (_depending on _flag) in the active editor  - updates View Menu

        """

        editor = self.getActiveEditor()

        if not editor:
            return

        editor.setEolVisibility(_flag)

    def showTabGuidelines(self, _flag):

        """

            slot - shows or hides Tab Guidelines (_depending on _flag) in the active editor  - updates View Menu

        """

        editor = self.getActiveEditor()

        if not editor:
            return

        editor.setIndentationGuides(_flag)

    def showLineNumbers(self, _flag):

        """

            slot - shows or hides line numbers (_depending on _flag) in the active editor  - updates View Menu

        """

        # QsciScintillaCustom linesChangedHandler sets margin width for line numbers
        editor = self.getActiveEditor()

        self.adjustLineNumbers(editor, _flag)

    def zoomIn(self):

        """

            slot - zooms in on all documents

        """
        self.zoomRange += 1

        self.configuration.setSetting("ZoomRange", self.zoomRange)

        for editor in self.getEditorList():
            editor.zoomTo(self.zoomRange)

    def zoomOut(self):

        """

            slot - zooms out on all documents

        """
        self.zoomRange -= 1

        self.configuration.setSetting("ZoomRange", self.zoomRange)

        for editor in self.getEditorList():
            editor.zoomTo(self.zoomRange)

    def save(self, _editor=None):

        """

            slot - saves current document

        """

        editor = self.getActiveEditor()

        file_name = ''

        if editor in self.getEditorList():
            file_name = self.getEditorFileName(editor)

        if file_name:
            return self.saveFile(file_name)

        return self.saveAs()

    def saveAs(self, suggestedName=None, _editor=None):

        """

            slot - implements save As... functionality

        """

        current_file_path = None

        current_extension = ""

        editor = None

        try:
            if _editor:
                editor = _editor

            else:
                editor = self.getActiveEditor()
            current_file_path = self.getEditorFileName(editor)

        except KeyError:

            pass

            # #adjusting line number margin width

            # if self.configuration.setting('DisplayLineNumbers') and editor.marginWidth(0):

            # self.adjustLineNumbers(editor,True)

        if current_file_path:

            file_split = os.path.splitext(str(current_file_path))

            current_extension = file_split[1]

            self.lastFileOpenPath = current_file_path

        elif self.lastFileOpenPath != '':

            current_file_path = self.lastFileOpenPath

        else:

            index = editor.panel.indexOf(editor)

            current_file_path = str(editor.panel.tabText(index))

            # else:

            # currentFilePath=self.lastFileOpenPath

        dbgMsg("suggestedName=", suggestedName)

        fileName = ""

        if suggestedName is None or isinstance(suggestedName,

                                               bool):  # saveAs is called by default with False bool type argument and we need to make sure that it is not converted to int index

            currentFilterString = self.getCurrentFilterString(current_extension)

            dbgMsg("currentFilterString=", currentFilterString)

            # currentFilterString="Text file (*.txt)"

            # behavior of file dialog is different on OSX appending preferred filter at the top does not really work , so we do not and the behavior is OK

            if sys.platform == 'darwin':

                fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", current_file_path,
                                                                    self.fileDialogFilters)

            else:

                fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", current_file_path,

                                                                    currentFilterString + ";;" + self.fileDialogFilters)

            dbgMsg("SAVE FILE NAME is:", fileName)

            if fileName == "":
                return False





        else:

            dbgMsg("")

            fileName = QtGui.QFileDialog.getSaveFileName(self, "Save File", suggestedName)

            if fileName == "":
                return False

        fileName = os.path.abspath(

            str(fileName))  # "normalizing" file name to make sure \ and / are used in a consistent manner

        activePanel = self.getActivePanel()

        if fileName:

            returnCode = self.saveFile(fileName)

            if returnCode:

                lexer = self.guessLexer(fileName)

                if lexer[0]:

                    activePanel.currentWidget().setLexer(lexer[0])

                    activePanel.currentWidget().setBraceMatching(lexer[3])



                else:  # lexer could not be guessed - use default lexer

                    activePanel.currentWidget().setLexer(None)

                tabIndex = activePanel.indexOf(activePanel.currentWidget())

                activePanel.setTabText(tabIndex, self.strippedName(fileName))

                self.commentStyleDict[activePanel.currentWidget()] = [lexer[1], lexer[

                    2]]  # associating comment style with the lexer

                currentEncoding = self.getEditorFileEncoding(activePanel.currentWidget())

                self.setPropertiesInEditorList(activePanel.currentWidget(), fileName, os.path.getmtime(str(fileName)),

                                               currentEncoding)

                self.setEditorProperties(activePanel.currentWidget())

            # before returning we check if du to saveAs some documents have been modified

            self.checkIfDocumentsWereModified()

            return returnCode

        return False

    def saveAll(self):

        """

            slot - saves all open documents  - if they need saving (e.g. were modified and changes were not saved)

        """

        """

        Resolving Issue #54:  Clicking 'Save All' jumps to main python script

        Link: https://github.com/CompuCell3D/CompuCell3D/issues/54

        Quick-Fix: Setting back the focus to current Index once all files are saved.

        TO-DO: Change the Save-All functionality

        """

        currentEditor = self.getCurrentEditor()

        currentIndex = currentEditor.panel.indexOf(currentEditor)

        unnamedFiles = {}

        for editor in self.getEditorList():

            if not self.getEditorFileName(editor) == '':

                index = editor.panel.indexOf(editor)

                editor.panel.setCurrentIndex(index)

                editor.setFocus(
                    Qt.MouseFocusReason)  # we have to set focus to editor so that it get's picked as active editor - this is the condition used by save functions

                if editor.isModified():

                    self.save()

                else:

                    editor.panel.setTabIcon(index, QtGui.QIcon(':/icons/document-clean.png'))



            else:

                index = editor.panel.indexOf(editor)

                unnamedFiles[editor] = editor.panel.tabText(index)

                # dealing with unnamed files:

        for editor in list(unnamedFiles.keys()):

            if editor == self.defaultEditor:  # we will not attempt to save content of the default editor. This editor should be removed anyway the moment we open any new file

                continue

            index = editor.panel.indexOf(editor)

            editor.panel.setCurrentIndex(index)

            editor.setFocus(Qt.MouseFocusReason)

            self.saveAs(unnamedFiles[editor])

        currentEditor.panel.setCurrentIndex(currentIndex)

        currentEditor.setFocus(Qt.MouseFocusReason)

    def about(self):

        """

            slot - displays about Twedit text

        """

        QtWidgets.QMessageBox.about(self,

                                    "About Twedit++5 - ver. {version}".format(version=__version__),

                                    "The <b>Twedit++5</b>  editor is a free Open-Source programmers editor\n"

                                    "Originally it was meant to be editor for Twitter and we limitted number of characters to 144\n"

                                    "However, after feedback from our users we were surprised to learn that people need more 144 characters to\n"

                                    "write software. We have since removed the limitation on number of characters... \n"

                                    "As a courtesy to our users no code written in this editor is catalogued by Google or any other data-mining company.\n"

                                    "<br><br>"

                                    "Copyright: Maciej Swat, <b>Swat International Productions, Inc.</b><br><br>"

                                    "Version {version}  rev. {revision} \n Commit tag: {commit_tag}".format(
                                        version=__version__, revision=__revision__, commit_tag=__commit_tag__)

                                    )

    def documentWasModified(self):

        self.setWindowModified(self.textEdit.document().isModified())

    def keySequenceStyleSpecifier(self, _qtStyleShortcut, _manualShortcut):

        # introdiced this to deal with PyQt4 bug on windows - whenever this bug gets fixed manual will be switched to False for all platforms

        manual = False

        if sys.platform.startswith('win'):
            manual = True

        if manual:

            return _manualShortcut

        else:

            return _qtStyleShortcut

    def updateRecentItemMenu(self, menuOwnerObj, _recentMenu, _recentItemSlot, _settingObj, _settingName):

        _recentMenu.clear()

        recentItems = _settingObj.setting(_settingName)

        print('setting_name=', _settingName)

        print('recentItems=', recentItems)

        itemCounter = 1

        for itemName in recentItems:
            actionText = '&%s %s' % (str(itemCounter), str(itemName))

            action = QAction("&%d %s " % (itemCounter, itemName), menuOwnerObj)

            _recentMenu.addAction(action)

            action.setData(QVariant(itemName))

            action.triggered.connect(_recentItemSlot)

            # menuOwnerObj.connect(action, SIGNAL("triggered()"), _recentItemSlot)

            # action.setData(QVariant(simulationFileName))

            itemCounter += 1

    def updateRecentDocumentsMenu(self):

        self.updateRecentItemMenu(self, self.recentDocumentsMenu, self.__loadRecentDocument, self.configuration,

                                  "RecentDocuments")

    def updateRecentDirectoriesMenu(self):

        self.updateRecentItemMenu(self, self.recentDirectoriesMenu, self.__openRecentDirectory, self.configuration,

                                  "RecentDirectories")

    def createActions(self):

        """

            fcn called in the constructor - it creates action associated with main window

        """

        self.newAct = QtWidgets.QAction(QtGui.QIcon(':/icons/document-new.png'), "&New",

                                        self, shortcut=self.keySequenceStyleSpecifier(QtGui.QKeySequence.New, 'Ctrl+N'),

                                        statusTip="Create a new file", triggered=self.newFile)

        am.addAction(self.newAct)

        self.openAct = QtWidgets.QAction(QtGui.QIcon(':/icons/document-open.png'),

                                         "&Open...", self,

                                         shortcut=self.keySequenceStyleSpecifier(QtGui.QKeySequence.Open, 'Ctrl+O'),

                                         statusTip="Open an existing file", triggered=self.open)

        am.addAction(self.openAct)

        self.saveAct = QtWidgets.QAction(QtGui.QIcon(':/icons/document-save.png'),

                                         "&Save", self,

                                         shortcut=self.keySequenceStyleSpecifier(QtGui.QKeySequence.Save, 'Ctrl+S'),

                                         statusTip="Save the document to disk", triggered=self.save)

        am.addAction(self.saveAct)

        self.saveAsAct = QtWidgets.QAction(QtGui.QIcon(':/icons/document-save-as.png'), "Save &As...", self,

                                           shortcut=self.keySequenceStyleSpecifier(QtGui.QKeySequence.SaveAs, ''),

                                           statusTip="Save the document under a new name",

                                           triggered=self.saveAs)

        am.addAction(self.saveAsAct)

        self.saveAllAct = QtWidgets.QAction(QtGui.QIcon(':/icons/document-save-all.png'), "Save All", self,

                                            shortcut="Ctrl+Shift+S",

                                            statusTip="Save all documents",

                                            triggered=self.saveAll)

        am.addAction(self.saveAllAct)

        self.printAct = QtWidgets.QAction(QtGui.QIcon(':/icons/document-print.png'), "Print...", self,

                                          shortcut="Ctrl+P",

                                          statusTip="Print current document", triggered=self.printCurrentDocument)

        am.addAction(self.printAct)

        self.closeAllAct = QtWidgets.QAction("Close All Tabs", self, statusTip="Close all tabs",

                                             triggered=self.__closeAll)

        am.addAction(self.closeAllAct)

        self.closeAllButCurrentAct = QtWidgets.QAction("Close All But Current Tab", self,

                                                       statusTip="Close all tabs except current one",

                                                       triggered=self.__closeAllButCurrent)

        am.addAction(self.closeAllButCurrentAct)

        self.renameAct = QtWidgets.QAction("Rename...", self, statusTip="Rename current document",

                                           triggered=self.__rename)

        am.addAction(self.renameAct)

        self.deleteDocAct = QtWidgets.QAction("Delete from disk", self, statusTip="Delete current document from disk",

                                              triggered=self.__deleteCurrentDocument)

        am.addAction(self.deleteDocAct)

        self.movetoOtherView = QtWidgets.QAction("Move To Other View", self, statusTip="Move document to new panel",

                                                 triggered=self.__moveToOtherView)

        am.addAction(self.movetoOtherView)

        self.exitAct = QtWidgets.QAction(QtGui.QIcon(':/icons/application-exit.png'), "E&xit", self, shortcut="Ctrl+Q",

                                         statusTip="Exit the application", triggered=self.close)

        am.addAction(self.exitAct)

        self.closeTabAct = QtWidgets.QAction(QtGui.QIcon(':/icons/tab-close.png'), "Close Tab", self, shortcut="Ctrl+W",

                                             statusTip="Close Current Tab", triggered=self.closeTab)

        am.addAction(self.closeTabAct)

        self.zoomInAct = QtWidgets.QAction(QtGui.QIcon(':/icons/zoom-in.png'), "Zoom In", self, shortcut="Ctrl+Shift+=",

                                           statusTip="Zoom In", triggered=self.zoomIn)

        am.addAction(self.zoomInAct)

        # self.zoomOutAct = QtWidgets.QAction(QtGui.QIcon(':/icons/zoom-out.png'), "Zoom Out", self, shortcut="Ctrl+-",

        #                                     statusTip="Zoom Out", triggered=self.zoomOut)

        self.zoomOutAct = QtWidgets.QAction(QtGui.QIcon(':/icons/zoom-out.png'), "Zoom Out", self, shortcut="Ctrl+-",

                                            statusTip="Zoom Out", triggered=self.zoomOut)

        am.addAction(self.zoomOutAct)

        self.foldAllAct = QtWidgets.QAction("Toggle Fold All", self, statusTip="Fold document at all folding points",

                                            triggered=self.foldAll)

        am.addAction(self.foldAllAct)

        self.wrapLinesAct = QtWidgets.QAction("Wrap Lines", self, statusTip="Wrap Lines in a current document")

        self.wrapLinesAct.setCheckable(True)

        am.addAction(self.wrapLinesAct)

        self.wrapLinesAct.triggered.connect(self.wrapLines)

        self.showWhitespacesAct = QtWidgets.QAction("Show Whitespaces", self,

                                                    statusTip="Show whitespaces in a current document")

        self.showWhitespacesAct.setCheckable(True)

        am.addAction(self.showWhitespacesAct)

        self.showWhitespacesAct.triggered.connect(self.showWhitespaces)

        self.showEOLAct = QtWidgets.QAction("Show EOL", self, statusTip="Show EOL in a current document")

        self.showEOLAct.setCheckable(True)

        am.addAction(self.showEOLAct)

        self.showEOLAct.triggered.connect(self.showEOL)

        self.showTabGuidelinesAct = QtWidgets.QAction("Show Tab Guidelines", self,

                                                      statusTip="Show Tab guidelines in a current document")

        self.showTabGuidelinesAct.setCheckable(True)

        am.addAction(self.showTabGuidelinesAct)

        self.showTabGuidelinesAct.triggered.connect(self.showTabGuidelines)

        self.showLineNumbersAct = QtWidgets.QAction("Show Line Numbers", self,

                                                    statusTip="Show line numbers in a current document")

        self.showLineNumbersAct.setCheckable(True)

        am.addAction(self.showLineNumbersAct)

        self.showLineNumbersAct.triggered.connect(self.showLineNumbers)

        self.showFindInFilesDockAct = QtWidgets.QAction("Show Find in Files Results", self,

                                                        statusTip="Show Find in Files Results",

                                                        triggered=self.toggleFindInFilesDock)

        self.showFindInFilesDockAct.setCheckable(True)

        am.addAction(self.showFindInFilesDockAct)

        self.blockCommentAct = QtWidgets.QAction("Block Comment/Uncomment", self, shortcut="Ctrl+/",

                                                 statusTip="Block Comment/Uncomment", triggered=self.block_comment)

        am.addAction(self.blockCommentAct)


        self.findAct = QtWidgets.QAction(QtGui.QIcon(':/icons/edit-find.png'), "Find...", self, shortcut="Ctrl+F",

                                         statusTip="Find...", triggered=self.find)

        am.addAction(self.findAct)

        self.findNextAct = QtWidgets.QAction("Find Next", self, shortcut="F3",

                                             statusTip="Find Next", triggered=self.findNextSimple)

        am.addAction(self.findNextAct)

        self.toggleBookmarkAct = QtWidgets.QAction(QtGui.QIcon(':/icons/flag.png'), "Toggle Bookmark", self,

                                                   shortcut="Alt+F2",

                                                   statusTip="Toggle Text Bookmark", triggered=self.toggleBookmark)

        am.addAction(self.toggleBookmarkAct)

        self.goToNextBookmarkAct = QtWidgets.QAction("Go To Next Bookmark", self, shortcut="F2",

                                                     statusTip="Go To Next Bookmark", triggered=self.goToNextBookmark)

        am.addAction(self.goToNextBookmarkAct)

        self.goToPreviousBookmarkAct = QtWidgets.QAction("Go To Previous Bookmark", self, shortcut="Shift+F2",

                                                         statusTip="Go To Previous Bookmark",

                                                         triggered=self.goToPreviousBookmark)

        am.addAction(self.goToPreviousBookmarkAct)

        self.deleteAllBookmarksAct = QtWidgets.QAction("Delete All Bookmarks", self, shortcut="",

                                                       statusTip="Delete All Bookmarks",

                                                       triggered=self.deleteAllBookmarks)

        am.addAction(self.deleteAllBookmarksAct)

        self.goToLineAct = QtWidgets.QAction("Go To Line...", self, shortcut="Ctrl+G",

                                             statusTip="Go To Line", triggered=self.goToLineShow)

        am.addAction(self.goToLineAct)

        self.goToMatchingBraceAct = QtWidgets.QAction("Go To Matching Brace", self, shortcut="Ctrl+]",

                                                      statusTip="Go To Matching Brace",

                                                      triggered=self.goToMatchingBrace)

        am.addAction(self.goToMatchingBraceAct)

        self.selectToMatchingBraceAct = QtWidgets.QAction("Select To Matching Brace", self, shortcut="Ctrl+Shift+]",

                                                          statusTip="Select To Matching Brace",

                                                          triggered=self.selectToMatchingBrace)

        am.addAction(self.selectToMatchingBraceAct)

        self.cutAct = QtWidgets.QAction(QtGui.QIcon(':/icons/edit-cut.png'), "Cu&t",

                                        self, shortcut=self.keySequenceStyleSpecifier(QtGui.QKeySequence.Cut, 'Ctrl+X'),

                                        statusTip="Cut the current selection's contents to the clipboard",

                                        triggered=self.cut)

        am.addAction(self.cutAct)

        self.copyAct = QtWidgets.QAction(QtGui.QIcon(':/icons/edit-copy.png'),

                                         "&Copy", self,

                                         shortcut=self.keySequenceStyleSpecifier(QtGui.QKeySequence.Copy, 'Ctrl+C'),

                                         statusTip="Copy the current selection's contents to the clipboard",

                                         triggered=self.copy)

        am.addAction(self.copyAct)

        self.pasteAct = QtWidgets.QAction(QtGui.QIcon(':/icons/edit-paste.png'),

                                          "&Paste", self,

                                          shortcut=self.keySequenceStyleSpecifier(QtGui.QKeySequence.Paste, 'Ctrl+V'),

                                          statusTip="Paste the clipboard's contents into the current selection",

                                          triggered=self.paste)

        am.addAction(self.pasteAct)

        self.increaseIndentAct = QtWidgets.QAction(QtGui.QIcon(':/icons/format-indent-more.png'), "Increase Indent",

                                                   self,

                                                   shortcut="Tab",

                                                   statusTip="Increase Indent", triggered=self.increaseIndent)

        am.addAction(self.increaseIndentAct)

        self.decreaseIndentAct = QtWidgets.QAction(QtGui.QIcon(':/icons/format-indent-less.png'), "Decrease Indent",

                                                   self,

                                                   shortcut="Shift+Tab",

                                                   statusTip="Decrease Indent", triggered=self.decreaseIndent)

        am.addAction(self.decreaseIndentAct)

        self.upperCaseAct = QtWidgets.QAction("Convert to UPPER case", self, shortcut="Ctrl+Shift+U",

                                              statusTip="Convert To Upper Case", triggered=self.convertToUpperCase)

        am.addAction(self.upperCaseAct)

        self.lowerCaseAct = QtWidgets.QAction("Convert to lower case", self, shortcut="Ctrl+U",

                                              statusTip="Convert To Lower Case", triggered=self.convertToLowerCase)

        am.addAction(self.lowerCaseAct)

        self.convertEOLAct = QtWidgets.QAction("Convert EOL", self, statusTip="Convert End of Line Character")

        self.convertEOLWinAct = QtWidgets.QAction("Windows EOL", self,

                                                  statusTip="Convert End of Line Character to Windows Style",

                                                  triggered=self.convertEolWindows)

        am.addAction(self.convertEOLWinAct)

        self.convertEOLUnixAct = QtWidgets.QAction("Unix EOL", self,

                                                   statusTip="Convert End of Line Character to Unix Style",

                                                   triggered=self.convertEolUnix)

        am.addAction(self.convertEOLUnixAct)

        self.convertEOLMacAct = QtWidgets.QAction("Mac OS 9 EOL", self,

                                                  statusTip="Convert End of Line Character to Mac Style",

                                                  triggered=self.convertEolMac)

        am.addAction(self.convertEOLMacAct)

        self.undoAct = QtWidgets.QAction(QtGui.QIcon(':/icons/edit-undo.png'), "Undo", self, shortcut="Ctrl+Z",

                                         statusTip="Undo", triggered=self.__undo)

        am.addAction(self.undoAct)

        self.redoAct = QtWidgets.QAction(QtGui.QIcon(':/icons/edit-redo.png'), "Redo", self, shortcut="Ctrl+Y",

                                         statusTip="Redo", triggered=self.__redo)

        am.addAction(self.redoAct)

        self.configurationAct = QtWidgets.QAction(QtGui.QIcon(':/icons/gear.png'), "Configure...", self, shortcut="",

                                                  statusTip="Configuration...", triggered=self.configurationUpdate)

        am.addAction(self.configurationAct)

        self.keyboardShortcutsAct = QtWidgets.QAction("Keyboard Shortcuts...", self, shortcut="",
                                                      statusTip="Reassign keyboard shortcuts",
                                                      triggered=self.keyboardShortcuts)

        am.addAction(self.keyboardShortcutsAct)

        self.reset_settings_act = QtWidgets.QAction("Reset Settings", self, shortcut="",
                                                      statusTip="Resets Settings to their default values",
                                                      triggered=self.reset_settings)

        am.addAction(self.reset_settings_act)

        self.switchToTabOnTheLeftAct = QtWidgets.QAction("Switch To Tab On The Left", self, shortcut="Ctrl+1",

                                                         statusTip="Switch To Tab On The Left",

                                                         triggered=self.switchToTabOnTheLeft)

        am.addAction(self.switchToTabOnTheLeftAct)

        self.switchToTabOnTheRightAct = QtWidgets.QAction("Switch To Tab On The Right", self, shortcut="Ctrl+2",

                                                          statusTip="Switch To Tab On The Right",

                                                          triggered=self.switchToTabOnTheRight)

        am.addAction(self.switchToTabOnTheRightAct)

        self.aboutAct = QtWidgets.QAction("&About", self,

                                          statusTip="Show the application's About box",

                                          triggered=self.about)

        am.addAction(self.aboutAct)

        self.aboutQtAct = QtWidgets.QAction("About &Qt", self,

                                            statusTip="Show the Qt library's About box",

                                            triggered=QApplication.instance().aboutQt)

        am.addAction(self.aboutQtAct)

        # self.panels[0].currentChanged.connect(self.tabIndexChanged_0) # connects tab changed signal to appropriate slot

        self.panels[0].tabCloseRequested.connect(self.closeTabIndex)

        self.panels[1].tabCloseRequested.connect(self.closeTabIndex)

        # self.panels[1].currentChanged.connect(self.tabIndexChanged_1)

        keyboardShortcutsDict = self.configuration.keyboardShortcuts()

        for actionName, keyboardShortcutText in keyboardShortcutsDict.items():
            am.setActionKeyboardShortcut(actionName, QKeySequence(keyboardShortcutText))

    def createMenus(self):

        """

            fcn called by constructor - creates menus of the main window

        """

        self.fileMenu = self.menuBar().addMenu("&File")

        self.fileMenu.addAction(am.actionDict["New"])

        self.fileMenu.addAction(am.actionDict["Open..."])

        self.fileMenu.addAction(am.actionDict["Save"])

        self.fileMenu.addAction(am.actionDict["Save As..."])

        self.fileMenu.addAction(am.actionDict["Save All"])

        self.fileMenu.addSeparator()

        # ---------------------------

        self.fileMenu.addAction(am.actionDict["Rename..."])

        self.fileMenu.addAction(am.actionDict["Close Tab"])

        self.fileMenu.addAction(am.actionDict["Close All Tabs"])

        self.fileMenu.addAction(am.actionDict["Close All But Current Tab"])

        self.fileMenu.addAction(am.actionDict["Delete from disk"])

        self.fileMenu.addSeparator()

        # ---------------------------

        self.fileMenu.addAction(am.actionDict["Print..."])

        self.fileMenu.addSeparator();

        # ---------------------------

        self.recentDocumentsMenu = self.fileMenu.addMenu("Recent Documents...")

        #         self.connect(self.recentDocumentsMenu , SIGNAL("aboutToShow()"), self.updateRecentDocumentsMenu)

        self.recentDocumentsMenu.aboutToShow.connect(self.updateRecentDocumentsMenu)

        self.recentDirectoriesMenu = self.fileMenu.addMenu("Recent Directories...")

        #         self.connect(self.recentDirectoriesMenu , SIGNAL("aboutToShow()"), self.updateRecentDirectoriesMenu)

        self.recentDirectoriesMenu.aboutToShow.connect(self.updateRecentDirectoriesMenu)

        self.fileMenu.addSeparator();

        # ---------------------------

        self.fileMenu.addAction(am.actionDict["Exit"])

        self.editMenu = self.menuBar().addMenu("&Edit")

        self.editMenu.addAction(am.actionDict["Copy"])

        self.editMenu.addAction(am.actionDict["Paste"])

        self.editMenu.addAction(am.actionDict["Cut"])

        self.editMenu.addSeparator()

        self.editMenu.addAction(am.actionDict["Block Comment/Uncomment"])

        self.editMenu.addAction(am.actionDict["Increase Indent"])

        self.editMenu.addAction(am.actionDict["Decrease Indent"])

        self.editMenu.addSeparator()

        self.editMenu.addAction(am.actionDict["Convert to UPPER case"])

        self.editMenu.addAction(am.actionDict["Convert to lower case"])

        self.convertEOLMenu = self.editMenu.addMenu("Convert EOL")

        self.convertEOLMenu.addAction(am.actionDict["Windows EOL"])

        self.convertEOLMenu.addAction(am.actionDict["Unix EOL"])

        self.convertEOLMenu.addAction(am.actionDict["Mac OS 9 EOL"])

        self.editMenu.addSeparator()

        self.editMenu.addAction(am.actionDict["Undo"])

        self.editMenu.addAction(am.actionDict["Redo"])

        self.searchMenu = self.menuBar().addMenu("&Search")

        self.searchMenu.addAction(am.actionDict["Find..."])

        self.searchMenu.addAction(am.actionDict["Find Next"])

        self.searchMenu.addSeparator()

        self.searchMenu.addAction(am.actionDict["Toggle Bookmark"])

        self.searchMenu.addAction(am.actionDict["Go To Next Bookmark"])

        self.searchMenu.addAction(am.actionDict["Go To Previous Bookmark"])

        self.searchMenu.addAction(am.actionDict["Delete All Bookmarks"])

        self.searchMenu.addSeparator()

        self.searchMenu.addAction(am.actionDict["Go To Line..."])

        self.searchMenu.addAction(am.actionDict["Go To Matching Brace"])

        self.searchMenu.addAction(am.actionDict["Select To Matching Brace"])

        self.viewMenu = self.menuBar().addMenu("&View")

        self.viewMenu.addAction(am.actionDict["Close Tab"])

        self.viewMenu.addAction(am.actionDict["Zoom In"])

        self.viewMenu.addAction(am.actionDict["Zoom Out"])

        self.fileMenu.addAction(am.actionDict["Switch To Tab On The Left"])

        self.fileMenu.addAction(am.actionDict["Switch To Tab On The Right"])

        self.viewMenu.addSeparator()

        # ---------------------------

        self.viewMenu.addAction(am.actionDict["Wrap Lines"])

        self.viewMenu.addAction(am.actionDict["Show Whitespaces"])

        self.viewMenu.addAction(am.actionDict["Show EOL"])

        self.viewMenu.addAction(am.actionDict["Show Tab Guidelines"])

        self.viewMenu.addAction(am.actionDict["Show Line Numbers"])

        self.viewMenu.addSeparator()

        # ---------------------------

        self.viewMenu.addAction(am.actionDict["Toggle Fold All"])

        self.viewMenu.addSeparator()

        # ---------------------------

        self.viewMenu.addAction(am.actionDict["Show Find in Files Results"])

        self.languageMenu = self.menuBar().addMenu("&Language")  # initialized in LanguageManager

        self.configurationMenu = self.menuBar().addMenu("&Configuration")

        self.configurationMenu.addAction(am.actionDict["Configure..."])

        self.configurationMenu.addAction(am.actionDict["Keyboard Shortcuts..."])
        # ---------------------------
        self.configurationMenu.addSeparator()
        self.configurationMenu.addAction(am.actionDict["Reset Settings"])

        self.menuBar().addSeparator()


        self.helpMenu = self.menuBar().addMenu("&Help")

        self.helpMenu.addAction(self.aboutAct)

        # self.helpMenu.addAction(self.aboutQtAct)

    def createToolBars(self):

        """

            fcn called by constructor - creates toolbars of the main window

        """

        self.toolBar = {}

        self.toolBar["File"] = self.addToolBar("File")

        self.toolBar["File"].setIconSize(QSize(32, 32))

        self.toolBar["File"].addAction(am.actionDict["New"])

        self.toolBar["File"].addAction(am.actionDict["Open..."])

        self.toolBar["File"].addAction(am.actionDict["Save"])

        self.toolBar["File"].addAction(am.actionDict["Save As..."])

        self.toolBar["File"].addAction(am.actionDict["Save All"])

        self.toolBar["Edit"] = self.addToolBar("Edit")

        self.toolBar["Edit"].addAction(am.actionDict["Copy"])

        self.toolBar["Edit"].addAction(am.actionDict["Paste"])

        self.toolBar["Edit"].addAction(am.actionDict["Cut"])

        self.toolBar["Edit"].addAction(am.actionDict["Increase Indent"])

        self.toolBar["Edit"].addAction(am.actionDict["Decrease Indent"])

        self.toolBar["Search"] = self.addToolBar("Search")

        self.toolBar["Search"].addAction(am.actionDict["Find..."])

        self.toolBar["Search"].addAction(am.actionDict["Toggle Bookmark"])

        self.toolBar["Configuration"] = self.addToolBar("Configurartion")

        self.toolBar["Configuration"].addAction(am.actionDict["Configure..."])

    #         for toolBarName, toolBar in self.toolBar.iteritems():

    #             print 'toolBarName=',toolBarName

    #             toolBar.setIconSize (self.toolbarIconSize)

    #         sys.exit()

    def createStatusBar(self):

        """

            fcn called by constructor - creates status bar of the main window

        """

        self.statusBar().showMessage("Ready")

    def toggleFindInFilesDock(self, _flag=False):

        """

            slot - shows or hides widget containing results of find in files - the hide event for this widget is updates status of View -> Show Find in Files Results action

        """

        dbgMsg("FLAG=", _flag)

        dbgMsg("self.showFindInFilesDockAct.isChecked():", self.showFindInFilesDockAct.isChecked())

        if self.findDock.isHidden():

            self.findDock.show()

        else:

            self.findDock.hide()

    def switchToTabOnTheLeft(self):

        activePanel = self.getActivePanel()

        currentIndex = activePanel.currentIndex()

        if currentIndex >= 1:
            activePanel.setCurrentIndex(currentIndex - 1)

    def switchToTabOnTheRight(self):

        activePanel = self.getActivePanel()

        currentIndex = activePanel.currentIndex()

        if currentIndex < activePanel.count() - 1:
            activePanel.setCurrentIndex(currentIndex + 1)

    def keyboardShortcuts(self):

        """

            slot - displays dialog for keyboard shortcut modifications

        """

        self.keyboardShortcutDlg = KeyboardShortcutsDlg(self, self)

        self.keyboardShortcutDlg.initializeShortcutTables()

        ret = self.keyboardShortcutDlg.exec_()

        if ret:
            self.keyboardShortcutDlg.reassignNewShortcuts()

    def reset_settings(self):
        """
        Resets settings to their default values
        :return:
        """
        self.configuration.reset_settings()

    def maybeSave(self, _editor=None):

        """

            fcn - checks if editor need to be saved

        """

        dbgMsg("slot maybeSave")

        # return True

        editor = None

        # print "_editor=",_editor

        if not _editor:

            editor = self.getActiveEditor()

        else:

            editor = _editor

            # if editor is None:

            # return

        dbgMsg("editor=", editor, " isModified()=", editor.isModified())

        if editor.isModified():

            fileName = ''

            if self.getEditorFileName(editor) != '':

                fileName = self.getEditorFileName(editor)

            else:

                index = editor.panel.indexOf(editor)

                fileName = editor.panel.tabText(index)

            message = "The document " + fileName + " has been modified.\nDo you want to save changes?"

            ret = QtWidgets.QMessageBox.warning(self, "Save Modification",

                                                message,

                                                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard |

                                                QtWidgets.QMessageBox.Cancel)

            if ret == QtWidgets.QMessageBox.Save:

                return self.save()

            elif ret == QtWidgets.QMessageBox.Cancel:

                return False

        return True

    def __undo(self):

        """

            slot - undo action for active editor

        """

        editor = self.getActiveEditor()

        editor.undo()

        self.checkActions()

    def __redo(self):

        """

            slot - redo action for active editor

        """

        editor = self.getActiveEditor()

        editor.redo()

        self.checkActions()

    def __closeAll(self):

        """

            slot - closes all the tabs

        """

        self.__closeAllButCurrent()

        self.closeTab()

    def __closeAllButCurrent(self):

        """

            slot - closes all the tabs except current one

        """

        currentEditor = self.getCurrentEditor()

        currentEditorTab = currentEditor.panel

        numberOfDocuments = currentEditorTab.count()

        removedTabs = 0

        for i in range(numberOfDocuments):

            editor = currentEditorTab.widget(i - removedTabs)

            if editor != currentEditor:
                self.closeTab(i - removedTabs)

                removedTabs += 1

        # closing remaining tabs

        for panel in self.panels:

            if panel != currentEditorTab:

                for i in range(panel.count()):
                    panel.widget(0).setFocus(Qt.MouseFocusReason)

                    self.closeTab(0, True, panel)

        self.activeTabWidget = currentEditorTab

    def __rename(self):

        """

            slot - renames current document - opens up save As dialog

        """

        fileName = self.getCurrentDocumentName()

        ret = self.saveAs()

        if ret:
            self.deleteDocument(fileName, False)  # don't display warning

    def deleteDocument(self, fileName, warningFlag=True):

        """

            deletes from hard drive document - fileName

        """

        if fileName == "":
            return

        fileName = os.path.abspath(fileName)  # normalize file name - jist in case

        if warningFlag:

            message = "You are about to completely delete " + fileName + " from disk.<br> " + "Proceed?"

            ret = QtWidgets.QMessageBox.information(self, "Delete from disk",

                                                    message,

                                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

            if ret == QtWidgets.QMessageBox.No:
                return

        try:

            os.remove(fileName)

            self.closeTab()

        except:

            message = "The document " + fileName + " cannot be deleted. Check if you have the right permissions."

            ret = QtWidgets.QMessageBox.information(self, "Delete from disk",

                                                    message,

                                                    QtWidgets.QMessageBox.Ok)

    def __deleteCurrentDocument(self):

        """

            slot - deletes current document from hard drive

        """

        fileName = self.getCurrentDocumentName()

        self.deleteDocument(fileName)

    def __moveToOtherView(self):

        """

            slot - moves one document from one panel to another

        """

        editor = self.clickedTabWidget.widget(self.clickedTabWidget.clickedTabPosition)

        tabIcon = self.clickedTabWidget.tabIcon(self.clickedTabWidget.indexOf(editor))

        tabText = self.clickedTabWidget.tabText(self.clickedTabWidget.indexOf(editor))

        sourceTabWidget = self.clickedTabWidget

        targetTabWidget = None

        swapTabs = False

        if self.clickedTabWidget == self.panels[0]:

            # swapTabs=True

            targetTabWidget = self.panels[1]

        else:

            targetTabWidget = self.panels[0]

            # if sourceTabWidget==self.panels[0] and sourceTabWidget.count()==1 and targetTabWidget.count()==0:

            # return

        editor.panel = targetTabWidget  # focusIn event will be called right after remove Tab so before that hapens we need to change tabAssignments in editor widgets

        sourceTabWidget.removeTab(sourceTabWidget.indexOf(editor))

        if targetTabWidget.isHidden():
            targetTabWidget.show()

            # if self.splitter.count()<2:

            # self.splitter.addWidget(self.panels[1])

        targetTabWidget.addTab(editor, tabIcon, tabText)

        targetTabWidget.setCurrentWidget(editor)

        editorTab = 0

        if editor.panel == self.panels[1]:
            editorTab = 1

        editor.setFocus(Qt.MouseFocusReason)

        # check it if it is ok to close extra tab widget

        if not sourceTabWidget.count():
            sourceTabWidget.hide()

        return

    def fileNameToClipboard(self):

        """

            slot - copies file name of current document to clipboard

        """

        clipboard = QApplication.clipboard()

        clipboard.setText(self.getCurrentDocumentName())

    def fileDirToClipboard(self):

        """

            slot - copies directory name of current document to clipboard

        """

        clipboard = QApplication.clipboard()

        clipboard.setText(os.path.dirname(self.getCurrentDocumentName()))

    def printCurrentDocument(self):
        """
            slot - prints current document - shows print dialog
        """

        editor = self.getActiveEditor()

        printer = PrinterTwedit()

        printDialog = QPrintDialog(printer, self)

        if printDialog.exec_() == QDialog.Accepted:
            dbgMsg("Paper size=", printer.paperSize())

            printer.setDocName(self.getCurrentDocumentName())

            printer.printRange(editor)

        dbgMsg("THIS IS PRINT CURRENT DOCUMENT")

    def convertToUpperCase(self):

        """

            slot - converts selected text to uppercase

        """

        editor = self.getActiveEditor()

        editor.SendScintilla(QsciScintilla.SCI_UPPERCASE)

    def convertToLowerCase(self):

        """

            slot - converts selected text to lowercase

        """

        editor = self.getActiveEditor()

        editor.SendScintilla(QsciScintilla.SCI_LOWERCASE)

    def convertEolWindows(self):

        """

            slot - converts EOL characters to windows style for current document

        """

        editor = self.getActiveEditor()

        editor.setEolMode(QsciScintilla.EolWindows)

        editor.convertEols(QsciScintilla.EolWindows)

    def convertEolUnix(self):

        """

            slot - converts EOL characters to Unix style for current document

        """

        editor = self.getActiveEditor()

        editor.setEolMode(QsciScintilla.EolUnix)

        editor.convertEols(QsciScintilla.EolUnix)

    def convertEolMac(self):

        """

            slot - converts EOL characters to Mac style for current document

        """

        editor = self.getActiveEditor()

        editor.setEolMode(QsciScintilla.EolMac)

        editor.convertEols(QsciScintilla.EolMac)

    def checkActions(self):

        """

            slot - updates state of actions (e.g. if undo is available after certain actions) for current editor

        """

        editor = self.getActiveEditor()

        if editor:

            self.undoAct.setEnabled(editor.isUndoAvailable())

            self.redoAct.setEnabled(editor.isRedoAvailable())

            if editor.wrapMode() == QsciScintilla.WrapNone:

                self.wrapLinesAct.setChecked(False)

            else:

                self.wrapLinesAct.setChecked(True)

            if editor.whitespaceVisibility() == QsciScintilla.SCWS_INVISIBLE:

                self.showWhitespacesAct.setChecked(False)

            else:

                self.showWhitespacesAct.setChecked(True)

            if editor.eolVisibility():

                self.showEOLAct.setChecked(True)

            else:

                self.showEOLAct.setChecked(False)

            if editor.indentationGuides():

                self.showTabGuidelinesAct.setChecked(True)

            else:

                self.showTabGuidelinesAct.setChecked(False)

            if editor.marginLineNumbers(0):  # checking if margin 0 (default for line numbers) is enabled

                self.showLineNumbersAct.setChecked(True)
                editor.line_numbers_enabled = True

            else:
                self.showLineNumbersAct.setChecked(False)
                editor.line_numbers_enabled = False

            self.languageManager.selectLexerBasedOnLexerObject(editor.lexer())

    def __modificationChanged(self, m):

        """

        Private slot to handle the modificationChanged signal.



        @param m modification status

        """

        dbgMsg(" INSIDE MODIFICATIONCHANGED SLOT")

        self.setWindowModified(m)

        self.checkActions()

    def maybeReload(self, tab):

        """

            slot called when focus is bought back to Twedit and one of the open documents has been modified externally

            It offer user opportunity to reload externally modified document

        """

        dbgMsg("slot maybeReload")

        message = "The document " + self.getEditorFileName(

            tab) + " has been modified by external program.\nDo you want to reload?"

        ret = QtWidgets.QMessageBox.warning(self, "Reload",

                                            message,

                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if ret == QtWidgets.QMessageBox.Yes:

            return self.reload_file(tab, self.getEditorFileName(tab))



        elif ret == QtWidgets.QMessageBox.No:

            return False

        return False


    def reload_file(self, editor: cc3d.twedit5.QsciScintillaCustom.QsciScintillaCustom, file_name: str):
        """
        reloads file
        :param editor:
        :param file_name:
        :return:
        """
        try:
            file = open(file_name, 'rb')
        except:

            QtWidgets.QMessageBox.warning(self, "Twedit++5",
                                          "Cannot read file %s:\n%s." % (file_name, "Check if the file is accessible"))
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        # have to disconnect signal which looks for text changes

        # editor.textChanged.disconnect(self.textChangedHandlers[editor].handleChangedText)

        editor.modificationChanged.disconnect(self.textChangedHandlers[editor].handleModificationChanged)

        editor.textChanged.disconnect(self.textChangedHandlers[editor].handleChangedText)

        editor.cursorPositionChanged.disconnect(self.handleCursorPositionChanged)

        txt, encoding = self.read_text_file(filename=file_name)

        file.close()

        # add re-read option to avoid duplicate reads in all cases?

        fh = codecs.open(file_name, 'rb', Encoding.normalizeEncodingName(encoding))

        txt = fh.read()

        fh.close()

        editor.setText(txt)

        # restore text-change watcher signal

        editor.modificationChanged.connect(self.textChangedHandlers[editor].handleModificationChanged)

        editor.textChanged.connect(self.textChangedHandlers[editor].handleChangedText)

        editor.cursorPositionChanged.connect(self.handleCursorPositionChanged)

        # "normalizing" file name to make sure \ and / are used in a consistent manner
        self.setPropertiesInEditorList(editor, os.path.abspath(str(file_name)), os.path.getmtime(str(file_name)),
                                       encoding)

        editor.setModified(False)

        self.updateTextSizeLabel()

        self.updateEncodingLabel()

        QtWidgets.QApplication.restoreOverrideCursor()

        return True

    def __loadRecentDocument(self):

        action = self.sender()

        fileName = ''

        if isinstance(action, QAction):
            # fileName = str(action.data().toString())

            fileName = action.data()

            self.loadFile(fileName)

    def loadFiles(self, fileNames):

        """

            loads files stores in fileNames list

        """

        self.deactivateChangeSensing = True

        for i in range(len(fileNames)):
            # "normalizing" file name to make sure \ and / are used in a consistent manner

            self.loadFile(os.path.abspath(str(fileNames[i])))

        self.deactivateChangeSensing = False

    def loadFile(self, fileName, _restoreFlag=False, _panel=-1):

        """

            loads single file (fielName) - accepts additional arguments like _restoreFlag or reference to panel into which load file

        """

        fileName = str(fileName)

        # "normalizing" file name to make sure \ and / are used in a consistent manner
        fileName = os.path.abspath(fileName)

        self.add_item_to_configuration_string_list(self.configuration, "RecentDocuments", fileName)

        # fileName = string.rstrip(fileName)  # remove extra trailing spaces - just in case
        fileName = fileName.rstrip()  # remove extra trailing spaces - just in case

        openFileDict = self.getFileNameToEditorWidgetMap()

        if fileName in list(openFileDict.keys()):

            # make tab with open file active

            try:

                editor = openFileDict[fileName]

                editor.panel.setCurrentWidget(editor)

                editor.setFocus(Qt.MouseFocusReason)

                modificationTimeEditor = self.getEditorFileModificationTime(editor)

                modificationTimeFile = os.path.getmtime(str(fileName))

                if modificationTimeEditor != modificationTimeFile:
                    self.maybeReload(editor)



            except KeyError as e:

                pass

            return

        file = None

        try:

            print('opening file ', fileName)

            file = open(fileName, 'r')

            print('DONE READING: file ', fileName)

        except:

            if not _restoreFlag:
                QtWidgets.QMessageBox.warning(self, "Twedit++5",

                                              "Cannot read file %s:\n%s." % (

                                                  fileName, "Check if the file is accessible"))

            return

            # inf = QtCore.QTextStream(file)

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        activeTab = self.panels[0]

        if _panel in (0, 1):

            activeTab = self.panels[_panel]

            self.panels[_panel].show()

        else:

            if self.activeTabWidget:

                activeTab = self.activeTabWidget

            else:

                for i in range(self.panels[0].count()):

                    editor = self.panels[0].widget(i)

                    # print "base tab i=",i," has focus=",editor.hasFocus()

                    if editor.hasFocus():
                        activeTab = self.panels[0]

                        break

                for i in range(self.panels[1].count()):

                    editor = self.panels[1].widget(i)

                    # print "extra tab i=",i," has focus=",editor.hasFocus()

                    if editor.hasFocus():
                        activeTab = self.panels[1]

                        break

        lexer = self.guessLexer(fileName)

        textEditLocal = None

        encoding = None

        if activeTab.count() == 1:

            # have to disconnect signal while opening document in first tab while new text is read in

            # activeTab.widget(0).textChanged.disconnect(self.textChangedHandlers[textEditLocal].handleChangedText)

            textEditLocal = activeTab.currentWidget()

            if not textEditLocal.isModified() and not textEditLocal.length():

                # it is better to create new QScintilla object than reuse old one

                # because reused editor tab had bookmar problems

                txt, encoding = self.read_text_file(filename=fileName)

                if not self.check_for_proper_text_file_encoding(encoding=encoding):
                    QtWidgets.QApplication.restoreOverrideCursor()
                    return

                activeTab.removeTab(0)

                textEditLocal = QsciScintillaCustom(self, activeTab)

                activeTab.addTab(textEditLocal, QtGui.QIcon(':/icons/document-clean.png'), self.strippedName(fileName))

                textEditLocal.setFocus(Qt.MouseFocusReason)

                textEditLocal = activeTab.currentWidget()
                #
                # txt, encoding = self.read_text_file(filename=fileName)
                #
                # if self.check_for_proper_text_file_encoding(encoding=encoding):

                textEditLocal.SendScintilla(QsciScintilla.SCI_SETCODEPAGE, QsciScintilla.SC_CP_UTF8)

                textEditLocal.setText(txt)

                # textEditLocal.setText(inf.readAll())

                activeTab.setTabText(0, self.strippedName(fileName))

                # self.setEditorProperties(textEditLocal)

                if lexer[0]:

                    textEditLocal.setLexer(lexer[0])

                    activeTab.currentWidget().setBraceMatching(lexer[3])

                    if self.configuration.setting("FoldText"):

                        textEditLocal.setFolding(QsciScintilla.BoxedTreeFoldStyle)

                    else:

                        textEditLocal.setFolding(QsciScintilla.NoFoldStyle)





            else:

                txt, encoding = self.read_text_file(filename=fileName)
                if not self.check_for_proper_text_file_encoding(encoding=encoding):
                    QtWidgets.QApplication.restoreOverrideCursor()
                    return

                textEditLocal = QsciScintillaCustom(self, activeTab)

                activeTab.addTab(textEditLocal, QtGui.QIcon(':/icons/document-clean.png'), self.strippedName(fileName))

                textEditLocal.setFocus(Qt.MouseFocusReason)

                self.setCurrentFile(fileName)

                # txt, encoding = self.read_text_file(filename=fileName)
                # if self.check_for_proper_text_file_encoding(encoding=encoding):
                textEditLocal.SendScintilla(QsciScintilla.SCI_SETCODEPAGE, QsciScintilla.SC_CP_UTF8)

                textEditLocal.setText(txt)

                if lexer[0]:

                    textEditLocal.setLexer(lexer[0])

                    activeTab.currentWidget().setBraceMatching(lexer[3])

                    if self.configuration.setting("FoldText"):

                        textEditLocal.setFolding(QsciScintilla.BoxedTreeFoldStyle)

                    else:

                        textEditLocal.setFolding(QsciScintilla.NoFoldStyle)
                activeTab.setTabIcon(0, QtGui.QIcon(':/icons/document-clean.png'))

                # enable textChangedSignal

                # activeTab.widget(0).textChanged.connect(self.textChangedHandlers[textEditLocal].handleChangedText)

        else:

            txt, encoding = self.read_text_file(filename=fileName)

            if not self.check_for_proper_text_file_encoding(encoding=encoding):
                QtWidgets.QApplication.restoreOverrideCursor()
                return

            textEditLocal = QsciScintillaCustom(self, activeTab)

            activeTab.addTab(textEditLocal, QtGui.QIcon(':/icons/document-clean.png'), self.strippedName(fileName))

            textEditLocal.setFocus(Qt.MouseFocusReason)

            self.setCurrentFile(fileName)

            # txt, encoding = self.read_text_file(filename=fileName)
            #
            # if self.check_for_proper_text_file_encoding(encoding=encoding):
            textEditLocal.SendScintilla(QsciScintilla.SCI_SETCODEPAGE, QsciScintilla.SC_CP_UTF8)

            textEditLocal.setText(txt)

            if lexer[0]:
                textEditLocal.setLexer(lexer[0])

                activeTab.currentWidget().setBraceMatching(lexer[3])

        self.setEditorProperties(textEditLocal)

        editor = textEditLocal

        activeTab.setCurrentWidget(editor)

        # loading document modifies its content but since no new text has been typed we set modified to false
        editor.setModified(False)

        editor.setWhitespaceVisibility(self.configuration.setting("DisplayWhitespace"))

        editor.setIndentationGuidesForegroundColor(self.indendationGuidesColor)

        editor.setIndentationGuides(self.configuration.setting("TabGuidelines"))

        #         editor.zoomTo(self.zoomRange) # we set zoom in setEditorproperties

        self.commentStyleDict[editor] = [lexer[1], lexer[2]]  # associating comment style with the lexer

        QtWidgets.QApplication.restoreOverrideCursor()

        self.setCurrentFile(fileName)

        self.setPropertiesInEditorList(editor, os.path.abspath(str(fileName)), os.path.getmtime(str(fileName)),

                                       encoding)  # file is associated with the tab,  "normalizing" file name to make sure \ and / are used in a consistent manner

        # adding text Changed Handler

        self.textChangedHandlers[editor] = ChangedTextHandler(editor, self)

        editorIndex = activeTab.indexOf(editor)

        dbgMsg("CONNECTING EDITOR INDEX=", editorIndex)

        activeTab.widget(editorIndex).modificationChanged.connect(

            self.textChangedHandlers[editor].handleModificationChanged)

        activeTab.widget(editorIndex).textChanged.connect(self.textChangedHandlers[editor].handleChangedText)

        activeTab.widget(editorIndex).cursorPositionChanged.connect(self.handleCursorPositionChanged)

        self.updateTextSizeLabel()

        self.updateEncodingLabel()

        dbgMsg(" SETTING fileName=", fileName, " os.path.getmtime(fileName)=", os.path.getmtime(str(fileName)))

        self.statusBar().showMessage("File loaded", 2000)
        if self.configuration.setting('DisplayLineNumbers'):

            self.adjustLineNumbers(activeTab.widget(editorIndex), True)
        else:
            self.adjustLineNumbers(activeTab.widget(editorIndex), False)

        self.checkActions()

    def check_for_proper_text_file_encoding(self, encoding):
        """
        checks if a file has a proper text encoding
        :param encoding: {str}
        :return:{bool}
        """
        if encoding is None:
            return False

        return encoding

    def read_text_file(self, filename):
        """
        Guesses encoding and reads correct file
        :param filename: {str}
        :return: text
        """

        fh, encoding = self.open_text_file(filename)

        txt = fh.read()

        fh.close()

        return txt, encoding

    def open_text_file(self, filename):
        """
        returns file handle fot he text file. USes codecs module to guess encoding
        :param filename: {str}
        :return: {file handle}
        """

        guessed_encoding = self.guess_encoding(filename=filename)

        # resorting to utf-8 if cannot guess encoding
        if guessed_encoding is None:
            guessed_encoding = 'utf-8'

        fh = codecs.open(filename, 'rb', guessed_encoding)

        return fh, guessed_encoding

    def guess_encoding(self, filename):

        """
        Guesses text encoding
        :param filename: {str}
        :return: {str} encoding
        """

        max_lines = -1

        if self.configuration.setting("EnableQuickTextDecoding"):
            max_lines = 1000

        guessed_encoding = decode(filename=filename, max_lines=max_lines)

        return guessed_encoding

    def add_item_to_configuration_string_list(self, setting_obj, setting_name, item_name, max_items=8):
        """
        Adds item (usually a path) to appropriate setting
        :param setting_obj:
        :param setting_name:
        :param item_name:
        :param max_items:
        :return:
        """

        string_list = setting_obj.setting(setting_name)

        number_of_items_to_remove = max(0, len(string_list) - max_items + 1)  # ideally this should be 1 or 0

        len_before = len(string_list)

        string_list = [item for item in string_list if item != item_name]

        len_after = len(string_list)

        removal_succesful = len_after < len_before

        if not removal_succesful:

            for i in range(number_of_items_to_remove):
                string_list = string_list[:-1]

        string_list.insert(0, item_name)

        string_list = remove_duplicates(string_list)

        setting_obj.setSetting(setting_name, string_list)

    def remove_item_from_configuration_string_list(self, setting_obj, setting_name, item_name):
        """
        Removes an item (usually a path) from appropriate setting
        :param setting_obj:
        :param setting_name:
        :param item_name:
        :return:
        """

        string_list = setting_obj.setting(setting_name)

        string_list = [item for item in string_list if item != item_name]

        string_list = remove_duplicates(string_list)

        setting_obj.setSetting(setting_name, string_list)

    def handleNewFocusEditor(self, _editor):

        """

            slot handling change of tab  or switching from panel to another

        """

        editor_tab = _editor

        # dbgMsg("self.fileDict=",self.fileDict)

        self.checkActions()
        self.updateTextSizeLabel()
        self.updateEncodingLabel()

        # have to do this check becaue adding tab triggers this slot but fileDict dictionary
        # has not been initialized yet
        if editor_tab in self.getEditorList():
            self.setCurrentFile(self.getEditorFileName(editor_tab))

    def guessLexer(self, _fileName):
        """
        guesses best lexer based on the file name . If canot guess the lexer returns default one
        """

        extension = ''

        file_split = os.path.splitext(str(_fileName))

        try:
            extension = file_split[1]
        except IndexError:
            pass

        if extension in list(self.extensionLanguageMap.keys()):

            try:
                return self.languageManager.languageLexerDictionary[self.extensionLanguageMap[extension]]
            except KeyError:
                pass

        # return format [lexer,begin comment, end comment, brace matching (0- nor matching, 1 matching), codeFolding]

        if os.path.basename(str(_fileName)).lower() == "cmakelists.txt":

            try:

                return self.languageManager.languageLexerDictionary["CMake"]

            except KeyError:

                pass

        if os.path.basename(str(_fileName)).lower() == "makefile":

            try:

                return self.languageManager.languageLexerDictionary["Makefile"]

            except KeyError:

                pass

        return [None, '', None, 0, 0, QsciScintilla.SCWS_INVISIBLE]

    def saveFile(self, _fileName, _editor=None):

        """

            saves _fileName

        """

        self.deactivateChangeSensing = True

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        text_edit_local = None

        if _editor:

            text_edit_local = _editor

        else:

            text_edit_local = self.getActiveEditor()

            # if self.configuration.setting('DisplayLineNumbers') and textEditLocal.marginWidth(0):

            # self.adjustLineNumbers(textEditLocal,True)

        active_tab = text_edit_local.panel

        encoding = 'utf-8'

        txt = str(text_edit_local.text())

        # work around glitch in scintilla: always make sure,

        # that the last line is terminated properly

        # eol = self.getLineSeparator()

        # if eol:

        # if len(txt) >= len(eol):

        # if txt[-len(eol):] != eol:

        # txt += eol

        # else:

        # txt += eol

        try:

            file_name_to_editor_map = self.getFileNameToEditorWidgetMap()

            editorLocal = None

            try:

                editorLocal = file_name_to_editor_map[_fileName]

            except:

                dbgMsg("COULD NOT FIND EDITOR FOR FILE NAME=", _fileName)

            try:

                encoding = self.getEditorFileEncoding(text_edit_local)

                # encoding=self.fileDict[textEditLocal][2]

            except:

                dbgMsg("COULD NOT FIND ENCODING")

            # encoding='ucs-2'

            txt, encoding = Encoding.encode(txt, encoding)

        except Encoding.CodingError as e:

            QtWidgets.QMessageBox.warning(self, "Twedit++5",

                                          "Cannot write file %s:\n%s." % (_fileName, repr(e)))

            self.deactivateChangeSensing = False

            return False

            # now write text to the file fn

        try:

            dbgMsg("SAVE - ENCODING = ", encoding, "\n\n\n\n\n")

            write_success = False

            if encoding == '':
                encoding = 'utf-8'

            """

             ISSUE #64: Save function saving a blank file in case of exception.

             FIX: Creating a .backup folder and creating backup of the file inside it before save operation

            """

            fileDirectory = os.path.dirname(_fileName)

            backupDirectory = os.path.join(fileDirectory, ".backup")

            backupFilePath = os.path.join(backupDirectory, os.path.basename(_fileName))

            if not os.path.exists(backupDirectory):
                os.makedirs(backupDirectory)

            if os.path.isfile(_fileName):
                shutil.copyfile(_fileName, backupFilePath)

            try:

                fh = codecs.open(_fileName + '~', 'wb', Encoding.normalizeEncodingName(encoding))

                # fh=open(_fileName, 'wb')

                Encoding.writeBOM(fh, encoding)

                fh.write(txt)

                write_success = True

            except:

                fh.close()

                # resorting to utf8 encoding

                encoding = 'utf-8'

                txt = str(text_edit_local.text())

                txt, encoding = Encoding.encode(txt, encoding)

                fh = codecs.open(_fileName + '~', 'wb', Encoding.normalizeEncodingName(encoding))

                try:

                    Encoding.writeBOM(fh, encoding)

                    fh.write(txt)

                    write_success = True

                except:

                    raise IOError('Could not save file using encoding %s' % encoding)



            finally:

                fh.close()

                if write_success:

                    try:

                        shutil.move(_fileName + '~', _fileName)

                    except shutil.Error:

                        # If there is error while renaming the newly saved file it will treated as write false

                        write_success = False

                        QtWidgets.QMessageBox.warning(self, "Twedit++",

                                                      "Cannot rename %s -> %s." % (_fileName + '~', _fileName))

                """

                In Finally block if the write_success is false then copy the backup file as a original file

                """

                if write_success == False:

                    shutil.copyfile(backupFilePath, _fileName)

                else:

                    try:

                        shutil.rmtree(backupDirectory)

                    except shutil.Error as e:

                        print('Could not remove backup directory {}'.format(backupDirectory))





        except IOError as why:

            QtWidgets.QApplication.restoreOverrideCursor()

            QtWidgets.QMessageBox.warning(self, "Twedit++",

                                          "Cannot write file %s:\n%s." % (_fileName, why))

            self.deactivateChangeSensing = False

            return False

        QtWidgets.QApplication.restoreOverrideCursor()

        dbgMsg("SAVE FILE EDITOR=", _editor, "\n\n\n\n")

        if _editor:

            # self.fileDict[_editor]=[_fileName,os.path.getmtime(_fileName),encoding] # saving new name and new modification time

            self.setPropertiesInEditorList(_editor, _fileName, os.path.getmtime(str(_fileName)),

                                           encoding)  # saving new name and new modification time

        else:

            self.setCurrentFile(_fileName)

            self.setPropertiesInEditorList(active_tab.currentWidget(), _fileName, os.path.getmtime(str(_fileName)),

                                           encoding)  # saving new name and new modification time

        self.statusBar().showMessage("File saved", 2000)

        if _editor:

            text_edit_local.setModified(False)

            index = active_tab.indexOf(text_edit_local)

            active_tab.setTabIcon(index, QtGui.QIcon(':/icons/document-clean.png'))

        else:

            active_tab.currentWidget().setModified(False)

            index = active_tab.indexOf(active_tab.currentWidget())

            active_tab.setTabIcon(index, QtGui.QIcon(':/icons/document-clean.png'))

        self.deactivateChangeSensing = False

        return True

    def setCurrentFile(self, fileName):

        """

            sets fileName as current filename and displays it in the title bar

        """

        self.curFile = fileName

        self.setWindowModified(False)

        if self.curFile:
            shown_name = self.curFile
        else:

            shown_name = 'untitled.txt'

        self.setWindowTitle(

            "[*]%s - Twedit++5" % shown_name)  # [*] is a placeholder for window modification flag - here it is positionwed before window title so that it is visible when windows has been modified

    def strippedName(self, fullFileName):

        """

            returns stripped file name - all path info is removed

        """

        return QtCore.QFileInfo(fullFileName).fileName()

    def __createDockWindow(self, name):

        """

        Private method to create a dock window with common properties.



        @param name object name of the new dock window (string or QString)

        @return the generated dock window (QDockWindow)

        """

        dock = QDockWidget(self)

        dock.setObjectName(name)

        # dock.setFeatures(QDockWidget.DockWidgetFeatures(QDockWidget.AllDockWidgetFeatures))

        return dock

    def __setupDockWindow(self, dock, where, widget, caption, showFlag=False):

        """

        Private method to configure the dock window created with __createDockWindow().

        

        @param dock the dock window (QDockWindow)

        @param where dock area to be docked to (Qt.DockWidgetArea)

        @param widget widget to be shown in the dock window (QWidget)

        @param caption caption of the dock window (string or QString)

        """

        if caption is None:
            caption = QString()

        self.addDockWidget(where, dock)

        dock.setWidget(widget)

        dock.hide()

        dock.setWindowTitle(caption)

        if showFlag:
            dock.show()
