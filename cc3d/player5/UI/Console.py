from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from cc3d.player5.CustomGui.CTabWidget import CTabWidget
from .ErrorConsole import ErrorConsole


class Console(CTabWidget):
    def __init__(self, parent):
        QTabWidget.__init__(self, parent)
        self.setTabPosition(QTabWidget.South)

        # self.__errorConsole.setText("Error: XML Error \n  File: cellsort_2D_error.xml\n
        # Line: 23 Col: 1 has the following problem not well-formed (invalid token) \n\n\n\n")

        self.__stdout = ConsoleWidget()
        self.__stdout.ensureCursorVisible()
        self.__stdoutIndex = self.addTab(self.__stdout, "Output")

        self.stdOutTextColor = QColor("black")
        self.stdErrTextColor = QColor("red")

        # self.__stderr = ConsoleWidget()
        # self.__stderr.ensureCursorVisible()
        # self.__stderrIndex = self.addTab(self.__stderr, self.trUtf8("Errors"))

        self.__errorConsole = ErrorConsole(self)
        self.__errorIndex = self.addTab(self.__errorConsole, "Errors")
        self.__menu = QMenu(self)
        self.__menu.addAction('Clear', self.__handleClear)
        self.__menu.addAction('Copy', self.__handleCopy)
        self.__menu.addSeparator()
        self.__menu.addAction('Select All', self.__handleSelectAll)

        self.setTabContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.__handleShowContextMenu)

        # self.setSizePolicy(
        #     QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        # self.connect(self,SIGNAL('customTabContextMenuRequested(const QPoint &, int)'),
        #              self.__handleShowContextMenu)

    def getStdErrConsole(self):
        return self.__stdout

    def getSyntaxErrorConsole(self):
        return self.__errorConsole

    def bringUpSyntaxErrorConsole(self):
        self.setCurrentWidget(self.__errorConsole)

    def bringUpOutputConsole(self):
        self.setCurrentWidget(self.__stdout)

    def __handleShowContextMenu(self, coord, index):
        """
        Private slot to show the tab context menu.
        
        @param coord the position of the mouse pointer (QPoint)
        @param index index of the tab the menu is requested for (integer)
        """
        self.__menuIndex = index
        coord = self.mapToGlobal(coord)
        self.__menu.popup(coord)

    def __handleClear(self):
        """
        Private slot to handle the clear tab menu entry.
        """
        self.widget(self.__menuIndex).clear()

    def __handleCopy(self):
        """
        Private slot to handle the copy tab menu entry.
        """
        self.widget(self.__menuIndex).copy()

    def __handleSelectAll(self):
        """
        Private slot to handle the select all tab menu entry.
        """
        self.widget(self.__menuIndex).selectAll()

    def showLogTab(self, tabname):
        """
        Public method to show a particular Log-Viewer tab.
        
        @param tabname string naming the tab to be shown ("stdout", "stderr")
        """
        if tabname == "stdout":
            self.setCurrentIndex(self.__stdoutIndex)
        # elif tabname == "stderr":
        # self.setCurrentIndex(self.__stderrIndex)
        else:
            raise RuntimeError("wrong tabname given")

    def appendToStdout(self, txt):
        """
        Public slot to appand text to the "stdout" tab.
        
        @param txt text to be appended (string or QString)
        """

        self.__stdout.setTextColor(self.stdOutTextColor)

        # self.__stdout.appendText(txt)
        self.__stdout.insertPlainText(txt)
        self.__stdout.ensureCursorVisible()

        # QApplication.processEvents() #this is causing application crash

    def appendToStderr(self, txt):
        """
        Public slot to appand text to the "stderr" tab.
        
        @param txt text to be appended (string or QString)
        """
        return
        # self.__stderr.appendText(txt)
        # QApplication.processEvents()

    # Changes the initial size of the console
    def sizeHint(self):
        return QSize(self.width(), 100)


class ConsoleWidget(QTextEdit):
    """
    Class providing a specialized text edit for displaying logging information.
    """

    def __init__(self, parent=None):
        """
        Constructor

        @param parent reference to the parent widget (QWidget)
        """
        QTextEdit.__init__(self, parent)
        self.setAcceptRichText(False)
        self.setLineWrapMode(QTextEdit.NoWrap)
        self.setReadOnly(True)
        self.setFrameStyle(QFrame.NoFrame)

        # Why do I need this? create the context menu
        self.__menu = QMenu(self)
        self.__menu.addAction('Clear', self.clear)
        self.__menu.addAction('Copy', self.copy)
        self.__menu.addSeparator()
        self.__menu.addAction('Select All', self.selectAll)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.__handleShowContextMenu)
        # self.connect(self, SIGNAL("customContextMenuRequested(const QPoint &)"),
        #     self.__handleShowContextMenu)
        #
        # self.setSizePolicy(
        #     QSizePolicy(QSizePolicy.Expanding,
        #                           QSizePolicy.Expanding))

    def __handleShowContextMenu(self, coord):
        """
        Private slot to show the context menu.

        @param coord the position of the mouse pointer (QPoint)
        """
        coord = self.mapToGlobal(coord)
        self.__menu.popup(coord)

    def appendText(self, txt):
        """
        Public method to append text to the end.

        @param txt text to insert (QString)
        """
        tc = self.textCursor()
        tc.movePosition(QTextCursor.End)
        self.setTextCursor(tc)
        self.insertPlainText(txt)
        self.ensureCursorVisible()
