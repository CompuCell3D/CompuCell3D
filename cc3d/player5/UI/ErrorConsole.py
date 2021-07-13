# todo - fix hide/open erro console
# class FindDock
import sys
from PyQt5.Qsci import *
from PyQt5.QtCore import *
from PyQt5 import QtCore

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import re
from PyQt5 import Qsci
from PyQt5 import QtGui

from cc3d.player5.Messaging import stdMsg, dbgMsg, pd, errMsg, setDebugging
# from QsciScintillaCustom import QsciScintillaCustom
# from PyQt4.Qsci.QsciScintilla import *
# NOTICE: complete scintilla documentation can be found here:
# http://www.scintilla.org/ScintillaDoc.htm
from . import CC3DSender
import os
import sys
from weakref import ref


# setDebugging(0)

class ErrorConsole(QsciScintilla):
    """
    Class providing a specialized text edit for displaying logging information.
    """
    # __pyqtSignals__ = ("closeCC3D()",)
    closeCC3D = pyqtSignal()

    # @QtCore.pyqtSignature("closeCC3D()")
    # @QtCore.pyqtSlot("closeCC3D()")
    def emitCloseCC3D(self):
        self.closeCC3D.emit()
        # self.emit(SIGNAL("closeCC3D()") )

    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget (QWidget) - here it is EditorWindow class
        """
        self.editorWindow = parent

        self.playerMainWidget = None

        QsciScintilla.__init__(self, parent)
        # # self.setFolding(5)
        self.setFolding(QsciScintilla.BoxedTreeFoldStyle)
        # self.setMarginSensitivity(3,True) 
        lexer = QsciLexerPython()
        dbgMsg(lexer.keywords(1), "\n\n\n\n")
        syntaxErrorLexer = SyntaxErrorLexer(self)
        self.setLexer(syntaxErrorLexer)

        self.setReadOnly(True)
        #         self.setReadOnly(False)

        self.setCaretLineVisible(True)
        # font=self.font()
        font = QFont("Courier New", 10)

        if sys.platform.startswith('da'):
            font = QFont("Courier New", 12)
        # font.setFixedPitch(True)
        self.setFont(font)

        self.setCaretLineBackgroundColor(QtGui.QColor('#E0E0F8'))  # current line has this color
        self.setSelectionBackgroundColor(
            QtGui.QColor('#E0E0F8'))  # any selection in the current line due to double click has the same color too
        # connecting SCN_DOUBLECLICK(int,int,int) to editor double-click
        # notice  QsciScintilla.SCN_DOUBLECLICK(int,int,int) is not the right name
        # self.connect(self, SIGNAL("SCN_DOUBLECLICK(int,int,int)"), self.onDoubleClick)

        GETFOLDLEVEL = QsciScintilla.SCI_GETFOLDLEVEL
        SETFOLDLEVEL = QsciScintilla.SCI_SETFOLDLEVEL
        HEADERFLAG = QsciScintilla.SC_FOLDLEVELHEADERFLAG
        LEVELBASE = QsciScintilla.SC_FOLDLEVELBASE
        NUMBERMASK = QsciScintilla.SC_FOLDLEVELNUMBERMASK
        WHITEFLAG = QsciScintilla.SC_FOLDLEVELWHITEFLAG

        headerLevel = LEVELBASE | HEADERFLAG
        lineStart = 1
        lineEnd = 3

        # self.SendScintilla(QsciScintilla.SCI_SETCARETSTYLE, QsciScintilla.CARETSTYLE_INVISIBLE) # make caret invisible

        self.lineNumberExtractRegex = re.compile('^[\s\S]*[L|l]ine:[\s]*([0-9]*)')
        self.colNumberExtractRegex = re.compile('^[\s\S]*[C|c]ol:[\s]*([0-9]*)')

        self.fileNameExtractRegex = re.compile('^[\s]*File:[\s]*([\S][\s\S]*)')

        # self.zoomRange=self.editorWindow.configuration.setting("ZoomRangeFindDisplayWidget")
        # self.zoomTo(self.zoomRange)

        dbgMsg("marginSensitivity=", self.marginSensitivity(0))

        self.cc3dSender = CC3DSender.CC3DSender(self)

    @property
    def editorWindow(self):
        try:
            o = self._editorWindow()
        except TypeError:
            o = self._editorWindow
        return o

    @editorWindow.setter
    def editorWindow(self, _i):
        try:
            self._editorWindow = ref(_i)
        except TypeError:
            self._editorWindow = _i

    @property
    def playerMainWidget(self):
        try:
            o = self._playerMainWidget()
        except TypeError:
            o = self._playerMainWidget
        return o

    @playerMainWidget.setter
    def playerMainWidget(self, _i):
        try:
            self._playerMainWidget = ref(_i)
        except TypeError:
            self._playerMainWidget = _i

    def setPlayerMainWidget(self, _playerMainWidget):

        self.playerMainWidget = _playerMainWidget

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
                # self.zoomRange+=1
            else:
                self.zoomOut()
                # self.zoomRange-=1                
                # self.editorWindow.configuration.setSetting("ZoomRangeFindDisplayWidget",self.zoomRange)
        else:
            # # calling wheelEvent from base class - regular scrolling
            super(QsciScintilla, self).wheelEvent(event)

    def clearAll(self):
        self.clear()
        self.setFolding(QsciScintilla.NoFoldStyle)  # stray fold character workaround

    def onMarginClick(self, _pos, _modifier, _margin):
        dbgMsg("_pos:", _pos, " modifier:", _modifier, " _margin:", _margin)
        lineClick = self.SendScintilla(QsciScintilla.SCI_LINEFROMPOSITION, _pos)
        dbgMsg("lineClick=", lineClick)
        levelClick = self.SendScintilla(QsciScintilla.SCI_GETFOLDLEVEL, lineClick)
        dbgMsg("levelClick=", levelClick)
        if levelClick & QsciScintilla.SC_FOLDLEVELHEADERFLAG:
            dbgMsg("Clicked Fold Header")
            self.SendScintilla(QsciScintilla.SCI_TOGGLEFOLD, lineClick)

    # to prevent QScintilla from selecting words on double click we implement mouseDoubleClisk event - 
    # in fact this is only necessary when using popup window (which we do)
    def mouseDoubleClickEvent(self, event):
        # self.setCursorPosition(0,0)                
        x = event.x()
        y = event.y()
        position = self.SendScintilla(QsciScintilla.SCI_POSITIONFROMPOINT, x, y)
        line = self.SendScintilla(QsciScintilla.SCI_LINEFROMPOSITION, position)
        self.onDoubleClick(position, line, None)
        event.accept()

    def onDoubleClick(self, _position, _line, _modifiers):
        dbgMsg("position=", _position, " line=", _line, " modifiers=", _modifiers)
        lineText = str(self.text(_line))
        dbgMsg("line text=", lineText)
        lineNumberGroups = self.lineNumberExtractRegex.search(lineText)
        lineNumber = -1
        colNumber = -1
        lineNumberWithFileName = -1
        fileName = ""
        try:
            if lineNumberGroups:
                lineNumber = int(lineNumberGroups.group(1))
                dbgMsg("Error at line=", lineNumber)
                lineNumberWithFileName = self.SendScintilla(QsciScintilla.SCI_GETFOLDPARENT, _line)
        except IndexError as e:

            dbgMsg("Line number not found")

        colNumberGroups = self.colNumberExtractRegex.search(lineText)
        try:
            if colNumberGroups:
                colNumber = int(colNumberGroups.group(1))
                dbgMsg("Error at column=", colNumber)

        except IndexError as e:

            dbgMsg("Col number not found")

        if lineNumberWithFileName >= 0:
            dbgMsg("THIS IS LINE WITH FILE NAME:")
            lineWithFileName = str(self.text(lineNumberWithFileName))

            dbgMsg(lineWithFileName)
            fileNameGroups = self.fileNameExtractRegex.search(lineWithFileName)
            print("fileNameGroups=", fileNameGroups)
            if fileNameGroups:
                try:
                    fileName = fileNameGroups.group(1)
                    fileName = fileName.strip()  # removing trailing white spaces
                    fileNameOrig = fileName  # store original file name as found by regex - it is used to locate unsaved files
                    # "normalizing" file name to make sure \ and / are used in a consistent manner
                    fileName1 = os.path.abspath(fileNameOrig)
                    # dbgMsg("FILE NAME :",fileName)
                    print("*******fileNameOrig=", fileNameOrig)
                    print("*******fileName=", fileName1)
                    # first we check if file exists and is readable:
                except IndexError as e:
                    dbgMsg("Could not extract file name")

        if fileName != "":
            self.cc3dSender.issueOpenFileRequest(fileName, int(lineNumber), int(colNumber))
            # self.selectAll()

    def findTabWithMatchingTabText(self, _tabText):
        """
            Looks for a tab in self.editorWindow with tab label matchin _tabText. returns this tab or None
        """

        for i in range(self.editorWindow.editTab.count()):

            currentTabText = str(self.editorWindow.editTab.tabText(i))
            # import string
            # currentTabText=string.rstrip(currentTabText)
            currentTabText = currentTabText.strip()  # removing trailing white spaces

            if currentTabText == _tabText:
                return self.editorWindow.editTab.widget(i)
        return None

    # Folding is sensitive to new lines - works fin if there is no new empty lines


# have to make sure that Search,Line, File at the beginning of the line do not have to include certain number fo leading spaces
# to make lexer work properly

# IMPORTANT. by default Lexer will process only visible text so if there is a state you want to pas from line to line you 
# have to store it in class variable or have a way to restore previous state  (here we use self.searchText) as a state variable to hold value of the searched text for each line. 

class SyntaxErrorLexer(QsciLexerCustom):
    def __init__(self, parent):
        Qsci.QsciLexerCustom.__init__(self, parent)
        self._styles = {
            0: 'Default',
            1: 'ErrorInfo',
            2: 'FileInfo',
            3: 'LineInfo',
            4: 'TextToFind'
        }
        for key, value in self._styles.items():
            setattr(self, value, key)

        self.editorWidget = parent
        self.colorizeEntireLineStates = [self.ErrorInfo, self.FileInfo]

        self.baseFont = QFont("Courier New", 10)
        if sys.platform.startswith('darwin'):
            self.baseFont = QFont("Courier New", 14)

        self.baseFont = Qsci.QsciLexerCustom.setDefaultFont(self, self.baseFont)
        self.baseFont = Qsci.QsciLexerCustom.defaultFont(self, 0)
        self.baseFont.setFixedPitch(True)

        self.baseFontBold = QFont(self.baseFont)
        self.baseFontBold.setBold(True)
        self.baseFontBold.setFixedPitch(True)
        self.searchText = ""

    @property
    def editorWidget(self):
        return self._editorWidget()

    @editorWidget.setter
    def editorWidget(self, _i):
        self._editorWidget = ref(_i)

    def description(self, style):
        return self._styles.get(style, '')

    # used by QsciLexer to style font colors
    def defaultColor(self, style):
        if style == self.Default:
            return QtGui.QColor('#000000')
        elif style == self.ErrorInfo:
            return QtGui.QColor('#FFFFFF')
        elif style == self.FileInfo:
            return QtGui.QColor('#04B404')
        elif style == self.LineInfo:
            return QtGui.QColor('#000000')
        elif style == self.TextToFind:
            return QtGui.QColor('#FF0000')

        return Qsci.QsciLexerCustom.defaultColor(self, style)

    # used by QsciLexer to style font properties
    def defaultFont(self, style):

        if style == self.Default:
            return self.baseFont
        elif style == self.ErrorInfo:
            # baseFont.setBold(True)
            return self.baseFontBold
        elif style == self.FileInfo:
            return self.baseFontBold

        return Qsci.QsciLexerCustom.defaultFont(self, style)

        # used by QsciLexer to style paper properties (background color)

    def defaultPaper(self, style):

        if style == self.Default:
            return QtGui.QColor('#FFFFFF')
        elif style == self.ErrorInfo:
            return QtGui.QColor('#DF0101')
        elif style == self.FileInfo:
            return QtGui.QColor('#E0F8E0')
        elif style == self.TextToFind:
            return QtGui.QColor('#F2F5A9')

        return Qsci.QsciLexerCustom.defaultPaper(self, style)

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
                source = str(editor.text()).encode(
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

        for line in source.splitlines(True):
            # todo - make sure this decoding is enought to convert bytearray to str
            line = line.decode('utf-8')
            length = len(line)
            # dbgMsg("line=",line)
            # dbgMsg(line)
            if line.startswith('\n'):
                style = self.Default
                dbgMsg("GOT EMPTY LINE")
                # sys.exit()
            else:
                if line.startswith('Error'):
                    state = self.ErrorInfo

                    # searchGroups =re.search('"([\s\S]*)"', line) # we have to use search instead of match - match matches onle beginning of the string , search searches through entire string


                    # # dbgMsg("searchGroups=",searchGroups)

                    # try: 

                    # if searchGroups:
                    # # dbgMsg(searchGroups.group(1))
                    # self.searchText=searchGroups.group(1)

                    # # dbgMsg("self.searchText=",self.searchText)
                    # except IndexError,e:
                    # self.searchText=""
                    # dbgMsg("COULD NOT EXTRACT TEXT")

                # elif line.startswith('  File'):
                elif line.startswith('  F'):
                    state = self.FileInfo

                # elif line.startswith('    Line'):    
                elif line.startswith('   '):

                    if self.searchText != "":
                        # dbgMsg("self.searchText=",self.searchText)
                        searchTextLength = len(self.searchText)
                        # pos = line.find(self.searchText)
                        # set_style(pos, self.LineInfo) # styling begining of the line
                        # set_style(searchTextLength, self.TextToFind) # styling searchText of the line
                        # length = length - pos - searchTextLength  # Default styling is applied to RHS
                        # state = self.ErrorInfo
                        # dbgMsg("LENGTH=",length)

                        # length = length - pos
                        # dbgMsg("line=",line)
                        startPos = 0
                        # string line is not use to output to the screen it is local to this fcn therefore it is safe to use lower
                        pos = line.lower().find(self.searchText.lower())
                        while pos != -1:
                            set_style(pos - startPos, self.LineInfo)  # styling begining of the line
                            set_style(searchTextLength, self.TextToFind)  # styling searchText of the line
                            startPos = pos + searchTextLength
                            pos = line.find(self.searchText, startPos)
                            state = self.LineInfo

                        state = self.LineInfo
                        length = length - startPos  # last value startPos if startPos point to the location right after last found searchText - to continue styling we tell lexer to style reminder of the line (length-startPos) with LineInfo style
                    else:
                        dbgMsg("DID NOT FIND SEARCH TEXT")
                        # state = self.Default
                        state = self.LineInfo

                        # # the following will style lines like "x = 0"

                        # pos = line.find('\tFile')

                        # if pos > 0:
                        # set_style(pos, self.ErrorInfo) #styling LHS pos is the length of styled text
                        # set_style(1, self.FileInfo)#styling = 1 is the length of styled text
                        # length = length - pos - 1
                        # state = self.ErrorInfo
                        # else:
                        # state = self.Default
                else:
                    # state = self.Default
                    state = self.LineInfo

            set_style(length, state)
            # folding implementation goes here
            headerLevel = LEVELBASE | HEADERFLAG
            # dbgMsg("HEADER LEVEL",headerLevel)

            # if index==0:
            if state == self.ErrorInfo:
                SCI(SETFOLDLEVEL, index, headerLevel)
            elif state == self.FileInfo:
                SCI(SETFOLDLEVEL, index,
                    headerLevel + 1)  # this subheader - inside header for ErrorInfo style - have to add +1 to folding level
            elif state == self.LineInfo:
                SCI(SETFOLDLEVEL, index,
                    LEVELBASE + 2)  # this is non-header fold line - since it is inside header level and headerLevel +1 i had to add +3 to the  LEVELBASE+2
            else:
                SCI(SETFOLDLEVEL, index,
                    LEVELBASE + 2)  # this is non-header fold line - since it is inside header level and headerLevel +1 i had to add +3 to the  LEVELBASE+2

            index += 1
