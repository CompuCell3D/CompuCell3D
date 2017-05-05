'''
TO DO:
* In the FindInFileResults display widget fi more than two occurences are found in the same line only the first one is highlighted 
'''
import re

from utils.global_imports import *
from utils.collection_utils import remove_duplicates

from Messaging import stdMsg, dbgMsg, pd, errMsg, setDebugging

# import ui_findandreplacedlg
import ui_findinfilesdlg

ALL_IN_FILES = 0
ALL_IN_ALL_OPEN_DOCS = 1
ALL_IN_CURRENT_DOC = 2

MAC = "qt_mac_set_native_menubar" in dir()


class FindInFilesResults:
    def __init__(self, _fileName="", _textToFind="", _unsavedFile=False):
        self.fileName = _fileName
        self.textToFind = _textToFind
        self.lineOccurences = []  # format of each list element is [lineNumber, lineText]
        self.lastLineAdded = -1
        self.totalHits = 0
        self.unsavedFile = _unsavedFile

    def addLineWithText(self, _lineNumber, _text):

        if self.lastLineAdded != _lineNumber:
            if not _text.endswith('\n'):
                # _text.append('\n')
                _text += '\n'
            # internally lines numbers start from 0    but we have to display them as starting from 1 - it is de-factor a convention for text editors            
            self.lineOccurences.append([_lineNumber + 1,
                                        _text])  # for display purposes we add 1 to _lineNumber i.e. we display line numbers as if they were counted from 1
            self.lastLineAdded = _lineNumber  # for accounting purposes lastLineNumber added is _lineNumber i.e. the one which is counted from 0

        self.totalHits += 1

    #         print 'totalHits=',self.totalHits

    def lastLineNumberAdded(self):
        return
        # this function will properly format find in files result so that they can be properly interpretted by the lexer

    # have to make lexer more robust though...
    def produceSummaryRepr(self, findInFilesResultsList, _textToFind=""):
        #         print 'lineOccurences=',findInFilesResultsList[0].lineOccurences
        #         print 'number of elements in lineOccurences=',len(findInFilesResultsList[0].lineOccurences)
        findInFileResultRepr = ""
        allHits = 0
        numberOfFiles = len(findInFilesResultsList)
        for findInFileResult in findInFilesResultsList:
            allHits += findInFileResult.totalHits
            #             print 'allHits=',allHits


            findInFileResultRepr += findInFileResult.__str__()
        hitsString = "hits"
        if allHits == 1:
            hitString = "hit"
        filesString = "files"
        if numberOfFiles == 1:
            filesString = "file"

        headerFindInFileResultRepr = "Search \"" + str(_textToFind) + "\" (" + str(
            allHits) + " " + hitsString + " in " + str(numberOfFiles) + " " + filesString + ")\n"

        return headerFindInFileResultRepr + findInFileResultRepr

    def __str__(self):
        # header="Search \""+str(self.textToFind)+"\"\n"
        # rep=header
        rep = ''
        if self.unsavedFile:
            rep = "  File: " + self.fileName  # "normalizing" file name to make sure \ and / are used in a consistent manner
        else:
            rep = "  File: " + os.path.abspath(
                unicode(self.fileName))  # "normalizing" file name to make sure \ and / are used in a consistent manner

        if self.totalHits == 1:
            rep += " (1 hit)"
        else:
            rep += " (" + str(self.totalHits) + " hits)"
        rep += "\n"
        for lineData in self.lineOccurences:
            rep += "    Line " + str(lineData[0]) + ":    " + str(lineData[1])
        return rep


class FindAndReplaceHistory:
    def __init__(self, _historyLength=20):
        self.findHistory = []
        self.replaceHistory = []
        self.historyLength = _historyLength
        self.textToFind = ""
        self.replaceText = ""
        self.wo = False  # words only
        self.re = False  # regular experessions flag
        self.cs = False  # case sensitive flag
        self.wrap = True  # start from the beginnig after reaching the end of text
        self.syntaxIndex = 0
        self.inSelection = False
        self.inAllSubfolders = False
        self.opacity = 75
        self.opacityOnLosingFocus = True
        self.opacityAlways = False
        self.transparencyEnable = True

        # self.findHistoryIF=QStringList()
        # self.replaceHistoryIF=QStringList()
        self.filtersHistoryIF = []
        self.directoryHistoryIF = []
        self.textToFindIF = ""
        self.replaceTextIF = ""
        self.filtersIF = ""
        self.directoryIF = ""

    def newSearchParameters(self, _text, _re, _cs, _wo, _wrap, _inSelection):
        flag = False
        if self.textToFind != _text:
            self.textToFind = _text
            # # # if str(self.textToFind).strip()!='':            
            self.findHistory.insert(0, self.textToFind)

            self.findHistory = list(set(self.findHistory))  # removing dupicates
            if len(self.findHistory) >= self.historyLength:
                self.findHistory.pop()  # removing last element that is over the limit

            # dbgMsg(self.findHistory)
            flag = True
        if self.re != _re:
            self.re = _re
            flag = True
        if self.cs != _cs:
            self.cs = _cs
            flag = True
        if self.wo != _wo:
            self.wo = _wo
            flag = True
        if self.wrap != _wrap:
            self.wrap = _wrap
            flag = True
        if self.inSelection != _inSelection:
            self.inSelection = _inSelection
            flag = True

        return flag

    def newSearchParametersIF(self, _text, _filters, _directory):
        flag = False
        if self.textToFindIF != _text:
            self.textToFindIF = _text
            self.findHistory.insert(0, self.textToFindIF)
            self.findHistory = remove_duplicates(self.findHistory)  # removing dupicates
            if len(self.findHistory) >= self.historyLength:

                self.findHistory.pop()  # removing last element that is over the limit

            flag = True

        if self.filtersIF != _filters:
            self.filtersIF = _filters
            if str(self.filtersIF).strip() != '':
                self.filtersHistoryIF.insert(0, self.filtersIF)
            print 'self.filtersHistoryIF=', len(self.filtersHistoryIF)

            self.filtersHistoryIF = remove_duplicates(self.filtersHistoryIF)
            print 'REMOVING DUPLICATES'
            print 'self.filtersHistoryIF=', len(self.filtersHistoryIF)
            for obj in self.filtersHistoryIF:
                print 'filter=', obj
            if len(self.filtersHistoryIF) >= self.historyLength:
                self.filtersHistoryIF.pop()


            flag = True


        if self.directoryIF != _directory:
            self.directoryIF = _directory
            if unicode(self.directoryIF).strip() != '':
                self.directoryHistoryIF.insert(0,self.directoryIF)
            self.directoryHistoryIF = remove_duplicates(self.directoryHistoryIF)
            if len(self.directoryHistoryIF) >= self.historyLength:
                self.directoryHistoryIF.pop()

            flag = True

        # for in files operations we always do search regardless if parameters change or not
        flag = True
        return flag

    def newReplaceParameters(self, _text, _replaceText, _re, _cs, _wo, _wrap, _inSelection):
        flag = self.newSearchParameters(_text, _re, _cs, _wo, _wrap, _inSelection)
        # pd("flag after newSearchParameters=",flag)
        # pd("self.replaceText=",self.replaceText," _replaceText=",_replaceText)
        if self.replaceText != _replaceText:
            self.replaceText = _replaceText
            # # # if str(self.replaceText).strip()!='':
            self.replaceHistory.insert(0,self.replaceText)
            self.replaceHistory = remove_duplicates(self.replaceHistory)
            if len(self.replaceHistory) >= self.historyLength:
                self.replaceHistory.pop()

            flag = True

        return flag

    def newReplaceParametersIF(self, _text, _replaceText, _filters, _directory):
        flag = self.newSearchParametersIF(_text, _filters, _directory)

        if self.replaceTextIF != _replaceText:
            self.replaceTextIF = _replaceText
            # # # if str(self.replaceTextIF).strip()!='':
            self.replaceHistory.insert(0, self.replaceTextIF)
            self.replaceHistory = remove_duplicates(self.replaceHistory)
            if len(self.replaceHistory) >= self.historyLength:
                self.replaceHistory.pop()


            flag = True

        return flag


class ClickToFocusEventFilter(QObject):
    def __init___(self, _qLineEdit):
        QObject.__init__(self)
        self.qLineEdit = _qLineEdit

    def eventFilter(self, obj, event):
        dbgMsg("INSIDE EVENT FILTER ", event.type())
        if event.type() == QEvent.FocusIn:
            self.qLineEdit.selectAll()
            # if event.key() == Qt.Key_Delete:
            # self.delkeyPressed.emit()
            # dbgMsg('delkey pressed')
            return True
        return False


class QLineEditCustom(QLineEdit):
    def __init__(self, _parent=None):
        QLineEdit.__init__(self, _parent)
        self.focusInCalled = False
        # have to monitor cursor position because qLineEdit inside QComboBox behaves weirdly - after each typed character the cursor is moved to the end of the word making it hard to type inside the line
        self.textEdited.connect(self.monitorCursor)
        self.cursorPositionChanged.connect(self.repositionCursor)
        #         self.returnPressed.connect(self.handleReturnPressed)
        self.pos = -1
        self.callbackFcn = None  # cannbackFcn to be called each time user presses enter provided callback is different than None

    # seems the solution is to eat MouseButtonPress and MouseButtonRelease events...
    # http://www.qtcentre.org/threads/10539-QLineEdit-selectAll%28%29-in-eventFilter
    # to highlight content of QLine edit we need to implement focusInEvent and also change behavior of mousePressEvent
    # when mousePressEvent is called right after focusIn we do nothing otherwise we use default mousePress Event


    def setReturnKeyCallbackFcn(self, _callback):
        self.callbackFcn = _callback

        # the approach with returnPressed signal  does not seem to work properly

    #     @pyqtSignature("")
    #     def handleReturnPressed(self):
    #         pass
    #         print 'THIS IS A STRING FROM THE LINE ',str(self.wgt.findLineEdit.text())
    #         if self.callbackFcn:
    #             self.callbackFcn()



    def mousePressEvent(self, event):  # this event handler is called second

        # dbgMsg("self.focusInCalled=",self.focusInCalled)

        if self.focusInCalled:
            self.focusInCalled = False
        else:
            QLineEdit.mousePressEvent(self, event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            if self.callbackFcn:
                self.callbackFcn()

            #             self.selectAll()
            event.accept()
        else:

            QLineEdit.keyPressEvent(self, event)

    def focusInEvent(self, event):  # this event handler is called first
        self.selectAll()
        self.focusInCalled = True
        QLineEdit.focusInEvent(self, event)

    def showEvent(self, event):  # this event handler is called first
        self.selectAll()
        self.focusInCalled = True
        QLineEdit.showEvent(self, event)

    def focusOutEvent(self, event):
        self.deselect()
        self.focusInCalled = False
        QLineEdit.focusOutEvent(self, event)

    def monitorCursor(self, _str):
        self.pos = self.cursorPosition()

    def repositionCursor(self, _oldPos, _newPos):
        # dbgMsg("repositionCursor=",(_oldPos,_newPos))
        if self.pos != -1:
            self.setCursorPosition(self.pos)
            self.pos = -1


class FindAndReplaceDlg(QDialog, ui_findinfilesdlg.Ui_FindInFiles):
    # signals
    searchingSignal = pyqtSignal(str)
    searchingAllInAllOpenDocsSignal = pyqtSignal(str, str, str, int)  # text, filters,directory, search mode
    replacingSignal = pyqtSignal(str, str)
    replacingAllSignal = pyqtSignal(str, str, bool)
    replacingAllInOpenDocsSignal = pyqtSignal(str, str, str, str, int)  # text,replaceText, filters,directory. replace mode
    searchingSignalIF = pyqtSignal(str, str, str)  # text, filters,directory
    replacingSignalIF = pyqtSignal(str, str, str, str)  # text,replaceText, filters,directory

    def __init__(self, text="", parent=None):
        super(FindAndReplaceDlg, self).__init__(parent)
        self.editorWindow = parent
        self.__text = unicode(text)
        self.__index = 0
        self.setupUi(self)

        self.findLineEdit = QLineEditCustom()

        self.findLineEdit.setReturnKeyCallbackFcn(
            self.on_findNextButton_clicked)  # to enable handling of the 'return' key pressed event
        self.findComboBox.setLineEdit(self.findLineEdit)
        self.findComboBox.setCompleter(None)  # we do not want autompletion here it is annoying
        # self.findComboBox.completer().setCaseSensitivity(1)

        self.replaceLineEdit = QLineEditCustom()
        self.replaceComboBox.setLineEdit(self.replaceLineEdit)
        self.replaceComboBox.setCompleter(None)  # we do not want autompletion here it is annoying
        # self.replaceComboBox.completer().setCaseSensitivity(1)

        # there are issues with Drawer dialog not getting focus when being displayed on linux
        # they are also not positioned properly so, we use "regular" windows 
        if sys.platform.startswith('win'):
            self.setWindowFlags(Qt.Drawer)  # dialogs without context help - only close button exists

        self.findAndReaplceHistory = None

        # IF stands for "in files"
        self.findLineEditIF = QLineEditCustom()
        self.findLineEditIF.setReturnKeyCallbackFcn(
            self.on_findAllButtonIF_clicked)  # to enable handling of the 'return' key pressed event
        self.findComboBoxIF.setLineEdit(self.findLineEditIF)
        self.findComboBoxIF.setCompleter(None)  # we do not want autompletion here it is annoying
        # self.findComboBoxIF.completer().setCaseSensitivity(1)
        self.replaceLineEditIF = QLineEditCustom()
        self.replaceComboBoxIF.setLineEdit(self.replaceLineEditIF)
        self.replaceComboBoxIF.setCompleter(None)  # we do not want autompletion here it is annoying
        # self.replaceComboBoxIF.completer().setCaseSensitivity(1)

        self.filtersLineEditIF = QLineEditCustom()
        self.filtersComboBoxIF.setLineEdit(self.filtersLineEditIF)
        self.filtersComboBoxIF.setCompleter(None)  # we do not want autompletion here it is annoying
        # self.filtersComboBoxIF.completer().setCaseSensitivity(1)

        self.directoryLineEditIF = QLineEditCustom()
        self.directoryComboBoxIF.setLineEdit(self.directoryLineEditIF)
        self.directoryComboBoxIF.setCompleter(None)  # we do not want autompletion here it is annoying
        # self.directoryComboBoxIF.completer().setCaseSensitivity(1)


        # self.findComboBox.setCompleter(0) # disallow word completion
        # synchronizing find and replace boxes on two tabs
        self.findLineEdit.textChanged.connect(self.findLineEditIF.setText)
        self.findLineEditIF.textChanged.connect(self.findLineEdit.setText)
        self.replaceLineEdit.textChanged.connect(self.replaceLineEditIF.setText)
        self.replaceLineEditIF.textChanged.connect(self.replaceLineEdit.setText)
        # synchronizing check boxes
        self.wholeCheckBox.toggled.connect(self.wholeCheckBoxIF.setChecked)
        self.wholeCheckBoxIF.toggled.connect(self.wholeCheckBox.setChecked)
        self.caseCheckBox.toggled.connect(self.caseCheckBoxIF.setChecked)
        self.caseCheckBoxIF.toggled.connect(self.caseCheckBox.setChecked)

        self.tabWidget.currentChanged.connect(self.tabChanged)
        self.alwaysRButton.toggled.connect(self.alwaysRButtonToggled)
        self.onLosingFocusRButton.toggled.connect(self.onLosingFocusRButtonToggled)

        self.transparencyGroupBox.toggled.connect(self.transparencyGroupBoxToggled)

        # synchronizing syntax boxes
        self.syntaxComboBox.activated.connect(self.syntaxComboBoxIF.setCurrentIndex)
        self.syntaxComboBoxIF.activated.connect(self.syntaxComboBox.setCurrentIndex)


        if not MAC:
            # # self.findButton.setFocusPolicy(Qt.NoFocus)
            self.replaceButton.setFocusPolicy(Qt.NoFocus)
            self.replaceAllButton.setFocusPolicy(Qt.NoFocus)
            self.closeButton.setFocusPolicy(Qt.NoFocus)
        self.updateUi()

        #         self.setWindowFlags(Qt.SubWindow|Qt.FramelessWindowHint |Qt.WindowSystemMenuHint |Qt.WindowStaysOnTopHint)
        if MAC:
            self.setWindowFlags(Qt.Tool | self.windowFlags())

    def setButtonsEnabled(self, _flag):
        # setEnabled on top widget blocks Qline edit keyboard focus on OSX . So instead we will enable each button individually
        # notice we do not touch close or clear history buttons
        self.findNextButton.setEnabled(_flag)
        self.findAllInOpenDocsButton.setEnabled(_flag)
        self.findAllInCurrentDocButton.setEnabled(_flag)
        self.replaceButton.setEnabled(_flag)
        self.replaceAllButton.setEnabled(_flag)
        self.replaceAllInOpenDocsButton.setEnabled(_flag)
        self.findAllButtonIF.setEnabled(_flag)
        self.replaceButtonIF.setEnabled(_flag)

    def tabChanged(self, idx):
        title = self.tabWidget.tabText(idx)
        dbgMsg("TITLE=", title)
        self.setWindowTitle(title)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:  # had to include it to ensure that on OSX find dialog closes after user presses ESC
            self.close()

    def changeEvent(self, event):

        if event.type() == QEvent.ActivationChange:
            if self.transparencyGroupBox.isChecked() and self.onLosingFocusRButton.isChecked():
                # opacity setting is often window manager dependent - e.g. on linux (KDE) you might need to enable desktop effects to enable transparency
                if self.isActiveWindow():

                    self.setWindowOpacity(1.0)
                else:
                    self.setWindowOpacity(self.transparencySlider.sliderPosition() / 100.0)

    def setFindAndReaplceHistory(self, _frh):
        self.findAndReaplceHistory = _frh

    def showEvent(self, event):
        self.findLineEdit.setFocus(True)
        self.initializeAllSearchLists(self.findAndReaplceHistory)
        if self.editorWindow.getCurrentEditor().hasSelectedText():
            self.findLineEdit.setText(self.editorWindow.getCurrentEditor().selectedText())

            # self.initializeSearchLists()

            ## this is quite strange but on windows there is no need to position dialog box, but on Linux, you have to do it manually
            # self.move(0,0) #always position popup at (0,0) then calculate required shift
            # self.adjustSize()


            ##setting position of the find dialog widget
            # geom=self.editorWindow.geometry()
            # pGeom=self.geometry()

            # pCentered_x=geom.x()+(geom.width()-pGeom.width())/2
            # pCentered_y=geom.y()+(geom.height()-pGeom.height())/2

            # self.move(pCentered_x,pCentered_y)

            # pGeom=self.cycleTabsPopup.geometry()

        QDialog.showEvent(self,
                          event)  # necesary to move dialog to the correct position - we do it only if we override show event

    def closeEvent(self, event):
        dbgMsg("CLOSE EVENT FOR FIND DIALOG")
        self.editorWindow.findDialogForm = None
        frh = self.findAndReaplceHistory
        frh.wo = self.wholeCheckBox.isChecked()
        frh.cs = self.caseCheckBox.isChecked()

        frh.syntaxIndex = self.syntaxComboBoxIF.currentIndex()

        frh.inSelection = self.inSelectionBox.isChecked()
        frh.inAllSubfolders = self.inAllSubFoldersCheckBoxIF.isChecked()

        frh.transparencyEnable = self.transparencyGroupBox.isChecked()
        frh.opacityOnLosingFocus = self.onLosingFocusRButton.isChecked()
        frh.opacityAlways = self.alwaysRButton.isChecked()
        frh.opacity = self.transparencySlider.sliderPosition()

    def initializeDialog(self, _frh):

        if _frh.cs:
            self.caseCheckBox.setChecked(True)
        else:
            self.caseCheckBox.setChecked(False)

        if _frh.wo:
            self.wholeCheckBox.setChecked(True)
        else:
            self.wholeCheckBox.setChecked(False)
        self.initializeAllSearchLists(_frh)

        # initializing syntax index
        self.syntaxComboBox.setCurrentIndex(_frh.syntaxIndex)
        self.syntaxComboBoxIF.setCurrentIndex(
            _frh.syntaxIndex)  # have to manually change current index both syntax combo boxes

        # initializing searchModifiers settings
        self.inSelectionBox.setChecked(_frh.inSelection)
        self.inAllSubFoldersCheckBoxIF.setChecked(_frh.inAllSubfolders)

        # initializing transparency settings
        self.transparencyGroupBox.setChecked(_frh.transparencyEnable)
        self.onLosingFocusRButton.setChecked(_frh.opacityOnLosingFocus)
        self.alwaysRButton.setChecked(_frh.opacityAlways)
        self.transparencySlider.setSliderPosition(_frh.opacity)

    def alwaysRButtonToggled(self, _flag):
        if self.transparencyGroupBox.isChecked() and _flag:
            self.setWindowOpacity(self.transparencySlider.sliderPosition() / 100.0)

    def onLosingFocusRButtonToggled(self, _flag):
        if _flag:
            self.setWindowOpacity(1.0)

    def transparencyGroupBoxToggled(self, _flag):
        if _flag:
            if self.alwaysRButton.isChecked():
                self.alwaysRButtonToggled(_flag)
        else:
            self.setWindowOpacity(1.0)

    def initializeSearchLists(self, _frh):
        # if self.replaceLineEdit.text()!=_frh.replaceHistory.first():
        # if self.replaceLineEdit.text()!='':
        # _frh.replaceHistory.prepend(self.replaceLineEdit.text())
        replaceTextToDisplayFirst = self.replaceLineEdit.text()
        replaceHistoryFirstItem = ''
        if len(_frh.replaceHistory):
            replaceHistoryFirstItem = _frh.replaceHistory[0]
        # pd("self.replaceLineEdit.text()=",self.replaceLineEdit.text()," _frh.replaceHistory.first()=",_frh.replaceHistory.first())
        if self.replaceLineEdit.text() != replaceHistoryFirstItem:
            if self.replaceLineEdit.text() != '':
                replaceTextToDisplayFirst = self.replaceLineEdit.text()

        self.findComboBox.clear()
        self.replaceComboBox.clear()
        self.findComboBoxIF.clear()
        self.replaceComboBoxIF.clear()

        self.findComboBox.addItems(_frh.findHistory)
        if replaceTextToDisplayFirst != '':
            self.findComboBox.addItem(replaceTextToDisplayFirst)

        self.replaceComboBox.addItems(_frh.replaceHistory)

        self.findComboBoxIF.addItems(_frh.findHistory)
        self.replaceComboBoxIF.addItems(_frh.replaceHistory)

    def initializeAllSearchLists(self, _frh):

        self.initializeSearchLists(_frh)
        self.directoryComboBoxIF.clear()
        self.filtersComboBoxIF.clear()
        self.directoryComboBoxIF.addItems(_frh.directoryHistoryIF)

        # print len(_frh.filtersHistoryIF)
        # print 
        # for strObj in _frh.filtersHistoryIF:
        # strObjPy=str(strObj)
        # strObjPy.strip()
        # if strObjPy!='':
        # self.filtersComboBoxIF.addItem(strObjPy)
        # print 'str=',strObj
        self.filtersComboBoxIF.addItems(_frh.filtersHistoryIF)

        # filtersTextToDisplayFirst=_frh.filtersHistoryIF.first()
        # print '_frh.filtersHistoryIF.first()=',_frh.filtersHistoryIF.first()

        # self.filtersLineEditIF.setText(filtersTextToDisplayFirst)
        # if filtersTextToDisplayFirst!='':            
        # self.filtersComboBoxIF.addItem(filtersTextToDisplayFirst)

    def findLineEditProcess(self, _text):
        dbgMsg("clicked Find Line Edit")

    # not including @pyqtSignature("") causes multiple calls to this fcn
    @pyqtSlot()
    def on_findNextButton_clicked(self):
        # dbgMsg("this is on findNext button clicked")
        self.searchingSignal.emit(str(self.findComboBox.lineEdit().text()))
        return

    @pyqtSlot()
    def on_findAllInOpenDocsButton_clicked(self):
        self.searchingAllInAllOpenDocsSignal.emit(str(self.findComboBox.lineEdit().text()), '', '',
                                                  ALL_IN_ALL_OPEN_DOCS)
        return

    @pyqtSlot()
    def on_findAllInCurrentDocButton_clicked(self):
        self.searchingAllInAllOpenDocsSignal.emit(str(self.findComboBox.lineEdit().text()), '', '', ALL_IN_CURRENT_DOC)
        return

    @pyqtSlot()
    def on_replaceAllInOpenDocsButton_clicked(self):
        self.replacingAllInOpenDocsSignal.emit(str(self.findComboBox.lineEdit().text()),
                                               str(self.replaceComboBox.lineEdit().text()), '', '',
                                               ALL_IN_ALL_OPEN_DOCS)
        return

    @pyqtSlot()
    def on_findCPB_clicked(self):
        # dbgMsg("this is on findNext button clicked"  )

        self.findAndReaplceHistory.findHistory = []
        self.findComboBox.clear()
        self.findComboBoxIF.clear()

    @pyqtSlot()
    def on_replaceCPB_clicked(self):
        dbgMsg("CLEAR REAPLCE HISTORY button clicked")
        self.findAndReaplceHistory.replaceHistory = []
        self.replaceComboBox.clear()
        self.replaceComboBoxIF.clear()

    @pyqtSlot()
    def on_filtersCPB_clicked(self):

        self.findAndReaplceHistory.filtersHistoryIF = []
        self.filtersComboBoxIF.clear()

    @pyqtSlot()
    def on_directoryCPB_clicked(self):

        self.findAndReaplceHistory.directoryHistoryIF = []
        self.directoryComboBoxIF.clear()

    @pyqtSlot()
    def on_replaceButton_clicked(self):
        # dbgMsg("this is on replace buttin clicked" )
        self.replacingSignal.emit(str(self.findComboBox.lineEdit().text()), str(self.replaceComboBox.lineEdit().text()))
        return

        dbgMsg("this is on replace buttin clicked")
        return
        regex = self.makeRegex()
        self.__text = regex.sub(unicode(self.replaceLineEdit.text()),
                                self.__text, 1)

    @pyqtSlot()
    def on_replaceAllButton_clicked(self):
        dbgMsg("this is on replaceAll buttin clicked")
        self.replacingAllSignal.emit(str(self.findComboBox.lineEdit().text()),
                                     str(self.replaceComboBox.lineEdit().text()), self.inSelectionBox.isChecked())
        return
        regex = self.makeRegex()
        self.__text = regex.sub(unicode(self.replaceLineEdit.text()),
                                self.__text)

    def updateUi(self):
        pass

    def text(self):
        return self.__text

    @pyqtSlot()
    def on_pickDirectoryButtonIF_clicked(self):
        dbgMsg("this is on pick directory button clicked")
        directory = QFileDialog.getExistingDirectory(self, "Look for files in the directory...")
        dbgMsg("directory=", directory)
        self.directoryLineEditIF.setText(directory)
        return

    @pyqtSlot()
    def on_findAllButtonIF_clicked(self):
        self.searchingSignalIF.emit(str(self.findComboBoxIF.lineEdit().text()),
                                    str(self.filtersComboBoxIF.lineEdit().text()).strip(),
                                    unicode(self.directoryComboBoxIF.lineEdit().text()).encode('UTF-8').strip())

        # w = QtWidgets.QPlainTextEdit()
        # w.setPlainText(self.directoryComboBoxIF.lineEdit().text())
        # dir_str = w.toPlainText().encode('utf-8').strip()
        # # unicode(self.directoryComboBoxIF.lineEdit().text()).encode('UTF-8').strip()
        #
        # self.searchingSignalIF.emit(str(self.findComboBoxIF.lineEdit().text()),
        #                             str(self.filtersComboBoxIF.lineEdit().text()).strip(),
        #                             dir_str)



    @pyqtSlot()
    def on_replaceButtonIF_clicked(self):
        self.replacingSignalIF.emit(str(self.findComboBoxIF.lineEdit().text()),
                                    str(self.replaceComboBoxIF.lineEdit().text()),
                                    str(self.filtersComboBoxIF.lineEdit().text()).strip(),
                                    unicode(self.directoryComboBoxIF.lineEdit().text()).strip())


# class FindDock
# import sys
# from PyQt4.Qsci import *
# from PyQt4 import Qsci
# from PyQt4 import QtGui
# from QsciScintillaCustom import QsciScintillaCustom
# from PyQt4.Qsci.QsciScintilla import *
# NOTICE: complete scintilla documentation can be found here:
# http://www.scintilla.org/ScintillaDoc.htm
class FindDisplayWidget(QsciScintilla):
    """
    Class providing a specialized text edit for displaying logging information.
    """

    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget (QWidget) - here it is EditorWindow class
        """
        self.editorWindow = parent
        QsciScintilla.__init__(self, parent)
        # # self.setFolding(5)
        self.setFolding(QsciScintilla.BoxedTreeFoldStyle)
        # self.setMarginSensitivity(3,True) 
        lexer = QsciLexerPython()
        dbgMsg(lexer.keywords(1), "\n\n\n\n")
        findInFilesLexer = FindInFilesLexer(self)
        self.setLexer(findInFilesLexer)

        self.setReadOnly(True)
        self.setCaretLineVisible(True)

        self.setCaretLineBackgroundColor(QtGui.QColor('#E0E0F8'))  # current line has this color
        self.setSelectionBackgroundColor(
            QtGui.QColor('#E0E0F8'))  # any selection in the current line due to double click has the same color too
        # connecting SCN_DOUBLECLICK(int,int,int) to editor double-click
        # notice  QsciScintilla.SCN_DOUBLECLICK(int,int,int) is not the right name
        # self.connect(self, SIGNAL("SCN_DOUBLECLICK(int,int,int)"), self.onDoubleClick)
        self.SCN_DOUBLECLICK.connect(self.onDoubleClick)

        GETFOLDLEVEL = QsciScintilla.SCI_GETFOLDLEVEL
        SETFOLDLEVEL = QsciScintilla.SCI_SETFOLDLEVEL
        HEADERFLAG = QsciScintilla.SC_FOLDLEVELHEADERFLAG
        LEVELBASE = QsciScintilla.SC_FOLDLEVELBASE
        NUMBERMASK = QsciScintilla.SC_FOLDLEVELNUMBERMASK
        WHITEFLAG = QsciScintilla.SC_FOLDLEVELWHITEFLAG

        headerLevel = LEVELBASE | HEADERFLAG
        lineStart = 1
        lineEnd = 3

        self.SendScintilla(QsciScintilla.SCI_SETCARETSTYLE, QsciScintilla.CARETSTYLE_INVISIBLE)  # make caret invisible

        self.lineNumberExtractRegex = re.compile('^[\s]*Line[\s]*([0-9]*)')
        self.fileNameWithSearchTextExtractRegex = re.compile('^[\s]*File:[\s]*([\S][\s\S]*)\(')

        self.zoomRange = self.editorWindow.configuration.setting("ZoomRangeFindDisplayWidget")
        self.zoomTo(self.zoomRange)

        dbgMsg("marginSensitivity=", self.marginSensitivity(0))

    def hideEvent(self, event):
        #         print "HIDE EVENT IN FIND DISPLAY WIDGET"
        self.editorWindow.showFindInFilesDockAct.setChecked(False)
        # self.editorWindow.showFindInFilesDockAct.trigger()

    def addNewFindInFilesResults(self, _str):
        self.setFolding(QsciScintilla.BoxedTreeFoldStyle)  # stray fold character workaround
        self.insertAt(_str, 0, 0)
        self.setCursorPosition(0, 0)
        # self.append(_str)

    # context menu handling
    def contextMenuEvent(self, event):
        menu = QMenu(self)
        copyAct = menu.addAction("Copy")
        selectAllAct = menu.addAction("Select All")
        clearAllAct = menu.addAction("Clear All")
        copyAct.triggered.connect(self.copy)
        selectAllAct.triggered.connect(self.selectAll)
        clearAllAct.triggered.connect(self.clearAll)

        menu.exec_(event.globalPos())

    def wheelEvent(self, event):
        if qApp.keyboardModifiers() == Qt.ControlModifier:
            # Forwarding wheel event to editor windowwheelEvent
            if event.delta() > 0:
                self.zoomIn()
                self.zoomRange += 1
            else:
                self.zoomOut()
                self.zoomRange -= 1
            self.editorWindow.configuration.setSetting("ZoomRangeFindDisplayWidget", self.zoomRange)
        else:
            # # calling wheelEvent from base class - regular scrolling
            super(QsciScintilla, self).wheelEvent(event)

    def clearAll(self):
        # self.clearFolds()
        self.clear()
        # SCI=self.SendScintilla
        # HEADERFLAG = QsciScintilla.SC_FOLDLEVELHEADERFLAG        
        # LEVELBASE = QsciScintilla.SC_FOLDLEVELBASE
        # SETFOLDLEVEL = QsciScintilla.SCI_SETFOLDLEVEL
        # headerLevel = LEVELBASE | HEADERFLAG
        # SCI(SETFOLDLEVEL, 0, headerLevel)

        # 
        # 
        self.setFolding(QsciScintilla.NoFoldStyle)  # stray fold character workaround
        # self.setFolding(QsciScintilla.BoxedTreeFoldStyle)

    def onMarginClick(self, _pos, _modifier, _margin):
        dbgMsg("_pos:", _pos, " modifier:", _modifier, " _margin:", _margin)
        lineClick = self.SendScintilla(QsciScintilla.SCI_LINEFROMPOSITION, _pos)
        dbgMsg("lineClick=", lineClick)
        levelClick = self.SendScintilla(QsciScintilla.SCI_GETFOLDLEVEL, lineClick)
        dbgMsg("levelClick=", levelClick)
        if levelClick & QsciScintilla.SC_FOLDLEVELHEADERFLAG:
            dbgMsg("Clicked Fold Header")
            self.SendScintilla(QsciScintilla.SCI_TOGGLEFOLD, lineClick)

    def onDoubleClick(self, _position, _line, _modifiers):

        dbgMsg("position=", _position, " line=", _line, " modifiers=", _modifiers)
        # print "position=",_position," line=",_line," modifiers=",_modifiers
        lineText = str(self.text(_line))

        dbgMsg("line text=", lineText)
        lineNumberGroups = self.lineNumberExtractRegex.search(lineText)
        lineNumber = -1
        lineNumberWithFileName = -1
        try:
            if lineNumberGroups:
                lineNumber = int(lineNumberGroups.group(1))
                lineNumber -= 1  # internally lines numbers start from 0    but we have to display them as starting from 1 - it is de-factor a convention for text editors
                dbgMsg("Searched text at line=", lineNumber)
                lineNumberWithFileName = self.SendScintilla(QsciScintilla.SCI_GETFOLDPARENT, _line)
        except IndexError, e:

            dbgMsg("Line number not found")

        if lineNumberWithFileName >= 0:
            dbgMsg("THIS IS LINE WITH FILE NAME:")
            lineWithFileName = unicode(self.text(lineNumberWithFileName))

            dbgMsg(lineWithFileName)
            fileNameGroups = self.fileNameWithSearchTextExtractRegex.search(lineWithFileName)
            if fileNameGroups:
                try:
                    fileNameWithSearchedText = fileNameGroups.group(1)
                    fileNameWithSearchedText = fileNameWithSearchedText.strip()  # removing trailing white spaces
                    fileNameWithSearchedTextOrig = fileNameWithSearchedText  # store original file name as found by regex - it is used to locate unsaved files
                    # "normalizing" file name to make sure \ and / are used in a consistent manner
                    fileNameWithSearchedText = os.path.abspath(fileNameWithSearchedText)
                    dbgMsg("FILE NAME WITH SEARCHED TEXT:", fileNameWithSearchedText)
                    # first we check if file exists and is readable:
                    currentEditorTab = None

                    try:
                        f = open(fileNameWithSearchedText, 'rb')
                        f.close()
                        self.editorWindow.loadFile(fileNameWithSearchedText)

                        activeTabWidget = self.editorWindow.getActivePanel()

                        currentEditorTab = activeTabWidget.currentWidget()

                        # currentEditorTab=activeTabWidget.currentWidget()
                        # print "OPENING FILE ",fileNameWithSearchedText
                        # print "current documernt name =", self.editorWindow.getCurrentDocumentName()                        
                    except IOError:
                        # in case file does not exist search for open tab with matching tab text                        
                        currentEditorTab = self.findTabWithMatchingTabText(fileNameWithSearchedTextOrig)

                    if not currentEditorTab:
                        return

                    # activeTabWidget.setCurrentWidget(currentEditorTab)    

                    currentEditorTab.setCursorPosition(lineNumber, 0)
                    currentEditorTab.ensureCursorVisible()
                    currentEditorTab.SendScintilla(QsciScintilla.SCI_SETFOCUS, True)
                except IndexError, e:
                    dbgMsg("Could not extract file name")

    def findTabWithMatchingTabText(self, _tabText):
        """
            Looks for a tab in self.editorWindow with tab label matchin _tabText. returns this tab or None
        """
        for tabWidget in self.editorWindow.panels:
            for i in range(tabWidget.count()):

                currentTabText = str(tabWidget.tabText(i))
                # import string
                # currentTabText=string.rstrip(currentTabText)
                currentTabText = currentTabText.strip()  # removing trailing white spaces

                if currentTabText == _tabText:
                    return tabWidget.widget(i)
        return None

    # lineWithFileNameText ="  File: C:/Program Files/twedit/tw edit.py (8 hits)"
    # lineWithFileNameGroups=re.search('^[\s]*File:[\s]*([\S][\s\S]*)\(',lineWithFileNameText)
    # dbgMsg(lineWithFileNameGroups)
    # if lineWithFileNameGroups:
    # dbgMsg("group 0=",lineWithFileNameGroups.group(0))
    # dbgMsg("group 1=",lineWithFileNameGroups.group(1)        )
    # figuring out the file name
    # using http://www.scintilla.org/ScintillaDoc.html#SCI_GETFOLDPARENT




    # self.foldLine(0)
    # self.foldAll()
    # def __handleShowContextMenu(self, coord):
    # """
    # Private slot to show the context menu.

    # @param coord the position of the mouse pointer (QPoint)
    # """
    # dbgMsg("Context Menu")
    # # coord = self.mapToGlobal(coord)
    # # self.__menu.popup(coord)


class CustomLexer(QsciLexerCustom):
    def __init__(self, parent):
        Qsci.QsciLexerCustom.__init__(self, parent)
        self._styles = {
            0: 'Default',
            1: 'Comment',
            2: 'Key',
            3: 'Assignment',
            4: 'Value',
        }
        for key, value in self._styles.iteritems():
            setattr(self, value, key)
        self.editorWidget = parent

    def description(self, style):
        return self._styles.get(style, '')

    def defaultColor(self, style):
        if style == self.Default:
            return QtGui.QColor('#000000')
        elif style == self.Comment:
            return QtGui.QColor('#C0C0C0')
        elif style == self.Key:
            return QtGui.QColor('#0000CC')
        elif style == self.Assignment:
            return QtGui.QColor('#CC0000')
        elif style == self.Value:
            return QtGui.QColor('#00CC00')
        return Qsci.QsciLexerCustom.defaultColor(self, style)

    def keywords(self, i):
        return "fcn"

    def styleText(self, start, end):
        editor = self.editor()
        if editor is None:
            return

        # scintilla works with encoded bytes, not decoded characters.
        # this matters if the source contains non-ascii characters and
        # a multi-byte encoding is used (e.g. utf-8)
        source = ''
        if end > editor.length():
            end = editor.length()
        if end > start:
            if sys.hexversion >= 0x02060000:
                # faster when styling big files, but needs python 2.6
                source = bytearray(end - start)
                editor.SendScintilla(
                    editor.SCI_GETTEXTRANGE, start, end, source)
            else:
                source = unicode(editor.text()
                                 ).encode('utf-8')[start:end]
        if not source:
            return

        # the line index will also be needed to implement folding
        index = editor.SendScintilla(editor.SCI_LINEFROMPOSITION, start)
        if index > 0:
            # the previous state may be needed for multi-line styling
            pos = editor.SendScintilla(
                editor.SCI_GETLINEENDPOSITION, index - 1)
            state = editor.SendScintilla(editor.SCI_GETSTYLEAT, pos)
        else:
            state = self.Default

        set_style = self.setStyling
        self.startStyling(start, 0x1f)

        # SCI = self.SendScintilla
        SCI = self.editorWidget.SendScintilla
        GETFOLDLEVEL = QsciScintilla.SCI_GETFOLDLEVEL
        SETFOLDLEVEL = QsciScintilla.SCI_SETFOLDLEVEL
        HEADERFLAG = QsciScintilla.SC_FOLDLEVELHEADERFLAG
        LEVELBASE = QsciScintilla.SC_FOLDLEVELBASE
        NUMBERMASK = QsciScintilla.SC_FOLDLEVELNUMBERMASK
        WHITEFLAG = QsciScintilla.SC_FOLDLEVELWHITEFLAG
        # scintilla always asks to style whole lines
        for line in source.splitlines(True):
            length = len(line)
            if line.startswith('#'):
                state = self.Comment
                dbgMsg("STYLING COMMENT")
            else:
                # the following will style lines like "x = 0"
                pos = line.find('=')
                if pos > 0:
                    set_style(pos, self.Key)  # styling LHS pos is the length of styled text
                    set_style(1, self.Assignment)  # styling = 1 is the length of styled text
                    length = length - pos - 1  # Value styling is applied to RHS
                    state = self.Value
                else:
                    state = self.Default
            set_style(length, state)
            # folding implementation goes here
            # headerLevel = LEVELBASE | HEADERFLAG
            # dbgMsg("HEADER LEVEL",headerLevel)

            # if index==0:
            # SCI(SETFOLDLEVEL, index, headerLevel)
            # else:
            # SCI(SETFOLDLEVEL, index, LEVELBASE+1)
            index += 1


# Folding is sensitive to new lines - works fin if there is no new empty lines
# have to make sure that Search,Line, File at the beginning of the line do not have to include certain number fo leading spaces 
# to make lexer work properly

# IMPORTANT. by default Lexer will process only visible text so if there is a state you want to pas from line to line you 
# have to store it in class variable or have a way to restore previous state  (here we use self.searchText) as a state variable to hold value of the searched text for each line. 

class FindInFilesLexer(QsciLexerCustom):
    def __init__(self, parent):
        Qsci.QsciLexerCustom.__init__(self, parent)
        self._states = {
            0: 'Default',
            1: 'SearchInfo',
            2: 'FileInfo',
            3: 'LineInfo',
            4: 'TextToFind'
        }

        for key, value in self._states.iteritems():
            setattr(self, value, key)

        self._styles = {
            'Default': 0,
            'SearchInfo': 1,
            'FileInfo': 2,
            'LineNumber': 3,
            'TextToFind': 4
        }
        for key, value in self._styles.iteritems():
            setattr(self, 'Style' + key, value)

        self.editorWidget = parent
        self.colorizeEntireLineStates = [self.SearchInfo, self.FileInfo]

        self.baseFont = Qsci.QsciLexerCustom.defaultFont(self, 0)
        self.baseFontBold = QFont(self.baseFont)
        self.baseFontBold.setBold(True)
        self.searchText = ""

        self.lineNumberRegex = re.compile('[\d]+')

    def language(self):
        return 'searchResult'  # correspoints to np++ style naming in  xml style-file

    def description(self, style):
        return self._styles.get(style, '')
        # #used by QsciLexer to style font colors
        # def defaultColor(self, style):
        # if style == self.Default:
        # return QtGui.QColor('#000000')
        # elif style == self.SearchInfo:
        # return QtGui.QColor('#0404B4')
        # elif style == self.FileInfo:
        # return QtGui.QColor('#04B404')
        # elif style == self.LineInfo:    
        # return QtGui.QColor('#000000')
        # elif style == self.TextToFind:
        # return QtGui.QColor('#FF0000')

        # return Qsci.QsciLexerCustom.defaultColor(self, style)

    # used by QsciLexer to style font properties
    def defaultFont(self, style):

        if style == self.Default:
            return self.baseFont
        elif style == self.SearchInfo:
            # baseFont.setBold(True)
            return self.baseFontBold
        elif style == self.FileInfo:
            return self.baseFontBold

        return Qsci.QsciLexerCustom.defaultFont(self, style)

        # #used by QsciLexer to style paper properties (background color)
        # def defaultPaper(self, style):

        # if style == self.Default:
        # return QtGui.QColor('#FFFFFF')
        # elif style == self.SearchInfo:            
        # return QtGui.QColor('#A9D0F5')
        # elif style == self.FileInfo:
        # return QtGui.QColor('#E0F8E0')
        # elif style == self.TextToFind:
        # return QtGui.QColor('#F2F5A9')

        # return Qsci.QsciLexerCustom.defaultPaper(self, style)    

    # To colorize entire line containit Search or File info we use  defaultEolFill fcn   
    def defaultEolFill(self, style):
        # This allowed to colorize all the background of a line.
        if style in self.colorizeEntireLineStates:
            return True
        return QsciLexerCustom.defaultEolFill(self, style)

    def styleText(self, start, end):

        editor = self.editor()
        if editor is None:
            return

        # scintilla works with encoded bytes, not decoded characters.
        # this matters if the source contains non-ascii characters and
        # a multi-byte encoding is used (e.g. utf-8)
        source = ''
        if end > editor.length():
            end = editor.length()
        if end > start:
            if sys.hexversion >= 0x02060000:
                # faster when styling big files, but needs python 2.6
                source = bytearray(end - start)
                editor.SendScintilla(
                    editor.SCI_GETTEXTRANGE, start, end, source)
            else:
                # source = unicode(editor.text()
                # ).encode('utf-8')[start:end]
                source = unicode(editor.text()).encode(
                    'utf-8')  # scanning entire text is way more efficient that doing it on demand especially when folding top level text (Search)
        if not source:
            return

        # the line index will also be needed to implement folding
        index = editor.SendScintilla(editor.SCI_LINEFROMPOSITION, start)
        if index > 0:
            # the previous state may be needed for multi-line styling
            pos = editor.SendScintilla(
                editor.SCI_GETLINEENDPOSITION, index - 1)
            state = editor.SendScintilla(editor.SCI_GETSTYLEAT, pos)
        else:
            state = self.Default

        set_style = self.setStyling
        self.startStyling(start, 0x1f)

        # SCI = self.SendScintilla
        SCI = self.editorWidget.SendScintilla
        GETFOLDLEVEL = QsciScintilla.SCI_GETFOLDLEVEL
        SETFOLDLEVEL = QsciScintilla.SCI_SETFOLDLEVEL
        HEADERFLAG = QsciScintilla.SC_FOLDLEVELHEADERFLAG
        LEVELBASE = QsciScintilla.SC_FOLDLEVELBASE
        NUMBERMASK = QsciScintilla.SC_FOLDLEVELNUMBERMASK
        WHITEFLAG = QsciScintilla.SC_FOLDLEVELWHITEFLAG
        # scintilla always asks to style whole lines

        previousLine = ''
        same_word_counter = 0

        for line in source.splitlines(True):
            length = len(line)
            # dbgMsg("line=",line)
            # dbgMsg(line)            
            if line.startswith('\n'):
                style = self.Default
                dbgMsg("GOT EMPTY LINE")
                # sys.exit()
            else:
                if line.startswith('Search'):
                    state = self.SearchInfo

                    searchGroups = re.search('"([\s\S]*)"',
                                             line)  # we have to use search instead of match - match matches onle beginning of the string , search searches through entire string

                    # dbgMsg("searchGroups=",searchGroups)

                    try:

                        if searchGroups:
                            # dbgMsg(searchGroups.group(1))
                            self.searchText = searchGroups.group(1)

                            # dbgMsg("self.searchText=",self.searchText)
                    except IndexError, e:
                        self.searchText = ""
                        dbgMsg("COULD NOT EXTRACT TEXT")



                # elif line.startswith('  File'):
                elif line.startswith('  F'):
                    state = self.FileInfo
                    # state=self.SearchInfo
                # elif line.startswith('    Line'):    
                elif line.startswith('   '):

                    state = self.SearchInfo

                    if self.searchText != "":
                        # dbgMsg("self.searchText=",self.searchText)
                        searchTextLength = len(self.searchText)
                        # pos = line.find(self.searchText)
                        # set_style(pos, self.LineInfo) # styling begining of the line
                        # set_style(searchTextLength, self.TextToFind) # styling searchText of the line
                        # length = length - pos - searchTextLength  # Default styling is applied to RHS
                        # state = self.SearchInfo
                        # dbgMsg("LENGTH=",length)

                        # length = length - pos
                        # dbgMsg("line=",line)
                        startPos = 0
                        # string line is not use to output to the screen it is local to this fcn therefore it is safe to use lower
                        pos = line.lower().find(self.searchText.lower())
                        # print 'line=',line
                        numberFinder = re.search(self.lineNumberRegex, line)
                        posNumStart = -1
                        posNumEnd = -1

                        if numberFinder:
                            posNumStart = numberFinder.start()
                            posNumEnd = numberFinder.end()
                            # print 'posNumStart,posNumEnd=',(posNumStart,posNumEnd)

                        # styling 'line' word    
                        set_style(posNumStart - startPos, self.StyleDefault)
                        # styling line number
                        set_style(posNumEnd - posNumStart, self.StyleLineNumber)
                        # state=self.Default                        
                        # length=length-posNumEnd # last value startPos if startPos point to the location right after last found searchText - to continue styling we tell lexer to style reminder of the line (length-startPos) with LineInfo style

                        # styling text between : and beginning of the first occurence of searched text
                        # # # set_style(pos-posNumEnd, self.StyleDefault)

                        # # # startPos=pos
                        # sometimes a single line will have multiple matched words . in that case we display multiple insrtances of this line coloring appropriate word find in consecutive lines 
                        if line == previousLine:
                            same_word_counter += 1
                        else:
                            same_word_counter = 0

                        startPos = posNumEnd
                        local_same_word_counter = 0
                        while pos != -1:
                            set_style(pos - startPos,
                                      self.Default)  # styling text between regular occurences of searched text
                            if local_same_word_counter == same_word_counter:
                                set_style(searchTextLength, self.TextToFind)  # styling searchText of the line
                            else:
                                set_style(searchTextLength, self.Default)  # styling searchText of the line

                            local_same_word_counter += 1

                            startPos = pos + searchTextLength
                            pos = line.find(self.searchText, startPos)

                            state = self.Default

                        state = self.Default
                        length = length - startPos  # last value startPos if startPos point to the location right after last found searchText - to continue styling we tell lexer to style reminder of the line (length-startPos) with LineInfo style

                    else:
                        dbgMsg("DID NOT FIND SEARCH TEXT")
                        state = self.Default

                else:
                    state = self.Default
                    # # # state = self.LineInfo
            # print 'state after while =',state     
            set_style(length, state)
            # folding implementation goes here
            headerLevel = LEVELBASE | HEADERFLAG
            # dbgMsg("HEADER LEVEL",headerLevel)

            # if index==0:
            if state == self.SearchInfo:
                SCI(SETFOLDLEVEL, index, headerLevel)
            elif state == self.FileInfo:
                SCI(SETFOLDLEVEL, index,
                    headerLevel + 1)  # this subheader - inside header for SearchInfo style - have to add +1 to folding level
            elif state == self.LineInfo:
                SCI(SETFOLDLEVEL, index,
                    LEVELBASE + 2)  # this is non-header fold line - since it is inside header level and headerLevel +1 i had to add +3 to the  LEVELBASE+2
            else:
                SCI(SETFOLDLEVEL, index,
                    LEVELBASE + 2)  # this is non-header fold line - since it is inside header level and headerLevel +1 i had to add +3 to the  LEVELBASE+2

            index += 1

            previousLine = line


class ConfigLexer(QsciLexerCustom):
    def __init__(self, parent):
        QsciLexerCustom.__init__(self, parent)
        self._styles = {
            0: 'Default',
            1: 'MultiLinesComment_Start',
            2: 'MultiLinesComment',
            3: 'MultiLinesComment_End',
            4: 'SingleLineComment'
        }
        for key, value in self._styles.iteritems():
            setattr(self, value, key)
        self._foldcompact = True
        self.__comment = [self.MultiLinesComment,
                          self.MultiLinesComment_End,
                          self.MultiLinesComment_Start,
                          self.SingleLineComment]

    def foldCompact(self):
        return self._foldcompact

    def setFoldCompact(self, enable):
        self._foldcompact = bool(enable)

    def language(self):
        return 'Config Files'

    def description(self, style):
        return self._styles.get(style, '')

    def defaultColor(self, style):
        if style == self.Default:
            return QColor('#000000')
        elif style in self.__comment:
            return QColor('#A0A0A0')
        return QsciLexerCustom.defaultColor(self, style)

    def defaultFont(self, style):
        if style in self.__comment:
            if sys.platform in ('win32', 'cygwin'):
                return QFont('Comic Sans MS', 9, QFont.Bold)
            return QFont('Bitstream Vera Serif', 9, QFont.Bold)
        return QsciLexerCustom.defaultFont(self, style)

    def defaultPaper(self, style):
        # Here we change the color of the background.
        # We want to colorize all the background of the line.
        # This is done by using the following method defaultEolFill() .
        if style in self.__comment:
            return QColor('#FFEECC')
        return QsciLexerCustom.defaultPaper(self, style)

    def defaultEolFill(self, style):
        # This allowed to colorize all the background of a line.
        if style in self.__comment:
            return True
        return QsciLexerCustom.defaultEolFill(self, style)

    def styleText(self, start, end):
        editor = self.editor()
        if editor is None:
            return

        SCI = editor.SendScintilla
        GETFOLDLEVEL = QsciScintilla.SCI_GETFOLDLEVEL
        SETFOLDLEVEL = QsciScintilla.SCI_SETFOLDLEVEL
        HEADERFLAG = QsciScintilla.SC_FOLDLEVELHEADERFLAG
        LEVELBASE = QsciScintilla.SC_FOLDLEVELBASE
        NUMBERMASK = QsciScintilla.SC_FOLDLEVELNUMBERMASK
        WHITEFLAG = QsciScintilla.SC_FOLDLEVELWHITEFLAG
        set_style = self.setStyling

        source = ''
        if end > editor.length():
            end = editor.length()
        if end > start:
            source = bytearray(end - start)
            SCI(QsciScintilla.SCI_GETTEXTRANGE, start, end, source)
        if not source:
            return

        compact = self.foldCompact()

        index = SCI(QsciScintilla.SCI_LINEFROMPOSITION, start)
        if index > 0:
            pos = SCI(QsciScintilla.SCI_GETLINEENDPOSITION, index - 1)
            prevState = SCI(QsciScintilla.SCI_GETSTYLEAT, pos)
        else:
            prevState = self.Default

        self.startStyling(start, 0x1f)

        for line in source.splitlines(True):
            # Try to uncomment the following line to see in the console
            # how Scintiallla works. You have to think in terms of isolated
            # lines rather than globally on the whole text.
            #            dbgMsg(line)

            length = len(line)
            # We must take care of empty lines.
            # This is done here.
            if length == 1:
                if prevState == self.MultiLinesComment or prevState == self.MultiLinesComment_Start:
                    newState = self.MultiLinesComment
                else:
                    newState = self.Default
                # We work with a non empty line.
            else:
                if line.startswith('/*'):
                    newState = self.MultiLinesComment_Start
                elif line.startswith('*/'):
                    if prevState == self.MultiLinesComment or prevState == self.MultiLinesComment_Start:
                        newState = self.MultiLinesComment_End
                    else:
                        newState = self.Default
                elif line.startswith('//'):
                    if prevState == self.MultiLinesComment or prevState == self.MultiLinesComment_Start:
                        newState = self.MultiLinesComment
                    else:
                        newState = self.SingleLineComment
                elif prevState == self.MultiLinesComment or prevState == self.MultiLinesComment_Start:
                    newState = self.MultiLinesComment
                else:
                    newState = self.Default

            set_style(length, newState)
            # Definition of the folding.
            # Documentation : http://scintilla.sourceforge.net/ScintillaDoc.html#Folding
            if newState == self.MultiLinesComment_Start:
                if prevState == self.MultiLinesComment:
                    level = LEVELBASE + 1
                else:
                    level = LEVELBASE | HEADERFLAG
            elif newState == self.MultiLinesComment or newState == self.MultiLinesComment_End:
                level = LEVELBASE + 1
            else:
                level = LEVELBASE

            SCI(SETFOLDLEVEL, index, level)

            pos = SCI(QsciScintilla.SCI_GETLINEENDPOSITION, index)
            prevState = SCI(QsciScintilla.SCI_GETSTYLEAT, pos)

            index += 1
